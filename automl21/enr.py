# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python3
# Implements elastic net regression (ENR) with CVXPY and ISTA
# with and without andersen acceleration as described in
# https://stanford.edu/~boyd/papers/pdf/scs_2.0_v_global.pdf

import torch
from torch import nn

import numpy as np
import numpy.random as npr

from scipy import io
import scipy.sparse as sp

import matplotlib.pyplot as plt
plt.style.use('bmh')

import cvxpy as cp

import hydra
import csv

from collections import namedtuple

import utils

import sys
sys.path.append('..')

from benchmark.accel.aa import AA
from benchmark.accel.neural_rec import NeuralLSTM, NeuralGRU

np.set_printoptions(precision=2)

dtype = torch.float32
torch.set_default_dtype(dtype)

import os, sys
if 'ipykernel' not in sys.modules and os.isatty(sys.stdout.fileno()):
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(
        mode='Plain', color_scheme='Neutral', call_pdb=1)


ENR_Instance = namedtuple("ENR_Instance", "A b A_np b_np x_hat x0 mu L")


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def soft_threshold(kappa, x):
    SX = x.sign() * torch.relu(x.abs() - kappa)
    return SX


def fixed_point_iteration(inst, alpha, x):
    kappa = alpha * inst.mu / 2.

    dx = inst.A.t().mv(inst.A.mv(x) - inst.b) + 0.5*inst.mu*x
    x_new = soft_threshold(kappa, x - alpha*dx)
    return x_new



class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        torch.manual_seed(cfg.seed)
        npr.seed(cfg.seed)

        iterate_size = cfg.n
        context_size = cfg.m*cfg.n + cfg.m
        self.aa = AA(iterate_size=iterate_size, memory_size=5)
        self.neural_accel = hydra.utils.instantiate(
            self.cfg.rec_model,
            iterate_size=iterate_size, context_size=context_size
        ).to(self.device)
        self.opt = torch.optim.Adam(
            self.neural_accel.parameters(), lr=self.cfg.lr)

        self.itr = 0
        self.loss_meter = utils.RunningAverageMeter(0.9)


    def solve_cvxpy(self, inst):
        # Solve the problem with CVXPY as a baseline we can take as
        # a near-ground-truth solution to validate the other algorithms.
        x = cp.Variable(self.cfg.n)
        obj = 0.5*cp.sum_squares(inst.A_np @ x - inst.b_np) + \
            inst.mu*(0.5*(1.-self.cfg.beta)*cp.sum_squares(x) + \
                     self.cfg.beta*cp.norm(x, 1))
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()
        x_star = torch.from_numpy(x.value).type_as(inst.A)
        return x_star


    def obj(self, inst, x):
        f = 0.5*(inst.A.mv(x) - inst.b).pow(2).sum() + \
            inst.mu*(0.5*(1.-self.cfg.beta)*x.pow(2).sum() + \
                     self.cfg.beta*x.norm(1))
        return f


    def residual(self, inst, alpha, x):
        x_new = fixed_point_iteration(inst, alpha, x)
        return x - x_new


    def solve_ISTA(self, inst, n_iter=10000, track_iterates=False):
        alpha = 1.8 / inst.L

        x = inst.x0.clone()
        g = self.residual(inst, alpha, x)
        g_init_norm = g.norm()
        if track_iterates:
            objs = [self.obj(inst, x)]
            rel_residual_norms = [1.]

        for k in range(n_iter):
            x = fixed_point_iteration(inst, alpha, x)
            g = self.residual(inst, alpha, x)
            if track_iterates:
                objs.append(self.obj(inst, x))
                rel_residual_norms.append(g.norm() / g_init_norm)

        d = {'solution': x.clone()}
        if track_iterates:
            d['objs'] =  torch.tensor(objs)
            d['rel_residual_norms'] = rel_residual_norms

        return d


    def solve_ISTA_AA(self, inst, n_iter=10000,
                      track_iterates=False):
        alpha = 1.8 / inst.L

        g = self.residual(inst, alpha, inst.x0)
        g_init_norm = g.norm()

        x, h = self.aa.init_instance(
            init_x = inst.x0.clone(), context=None)

        if track_iterates:
            objs = [self.obj(inst, x)]
            rel_residual_norms = [1.]

        for k in range(n_iter):
            prev_x = x.clone()
            x = fixed_point_iteration(inst, alpha, x)
            try:
                x, g, h = self.aa.update(fx=x, x=prev_x, hidden=h)
            except:
                pass

            if track_iterates:
                objs.append(self.obj(inst, x))
                rel_residual_norms.append(g.norm() / g_init_norm)

        d = {'solution': x.clone()}
        if track_iterates:
            d['objs'] =  torch.tensor(objs)
            d['rel_residual_norms'] = rel_residual_norms
        return d


    def solve_ISTA_neural(self, inst, n_iter=10000, track_iterates=False):
        alpha = 1.8 / inst.L

        g = self.residual(inst, alpha, inst.x0)
        g_init_norm = g.norm()

        context = torch.cat((inst.A.reshape(-1), inst.b))
        x, h = self.neural_accel.init_instance(
            init_x = inst.x0.clone(), context=context)

        losses = []

        if track_iterates:
            objs = [self.obj(inst, x)]
            rel_residual_norms = [1.]

        for k in range(n_iter):
            prev_x = x.clone()
            x = fixed_point_iteration(inst, alpha, x)
            x, g, h = self.neural_accel.update(fx=x, x=prev_x, hidden=h)

            losses.append(g.norm() / g_init_norm)

            if track_iterates:
                objs.append(self.obj(inst, x))
                rel_residual_norms.append(g.norm() / g_init_norm)

        loss = sum(losses) / len(losses)

        d = {
            'solution': x.clone(),
            'loss': loss,
        }
        if track_iterates:
            d['objs'] =  torch.tensor(objs)
            d['rel_residual_norms'] = rel_residual_norms

        return d


    def sample_single_inst(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            npr.seed(seed)

        A = torch.randn(self.cfg.m, self.cfg.n, device=self.device)
        x_hat = torch.from_numpy(
            sp.random(self.cfg.n, 1, density=self.cfg.sparse_density,
                      data_rvs=npr.randn).todense()
        ).type_as(A).squeeze().to(self.device)

        w = torch.randn(self.cfg.m, device=self.device)
        b = A.mv(x_hat) + 0.1*w

        x0 = torch.randn(self.cfg.n, device=self.device)
        x0 = x0 / x0.norm()

        mu_max = (A.t().mv(b)).abs().max().item()
        mu = 0.001*mu_max

        ATA = A.t().mm(A)
        L = torch.eig(ATA).eigenvalues[:,0].max() + mu/2.

        A_np = to_np(A)
        b_np = to_np(b)
        return ENR_Instance(A, b, A_np, b_np, x_hat, x0, mu, L)


    def run(self):
        self.init_logging()
        while self.itr < self.cfg.num_model_updates:
            inst = self.sample_single_inst(seed=self.itr)
            d = self.solve_ISTA_neural(
                inst, n_iter=self.cfg.num_ISTA_iterations_train,
            )
            loss = d['loss']
            self.loss_meter.update(loss.item())


            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.neural_accel.parameters(), self.cfg.max_gradient_norm)
            # self.model.print_grad_stats()
            self.opt.step()

            if self.itr % self.cfg.test_freq == 0:
                print(f'\n=== iter {self.itr}, train loss={self.loss_meter.avg:.2e}\n')
                self.writer.writerow({
                    'iter': self.itr,
                    'loss': self.loss_meter.avg,
                })
                self.logf.flush()

                self.plot_single(
                    n_iter=self.cfg.num_ISTA_iterations_eval,
                    tag=f'{self.itr:06d}')

                self.plot_agg(
                    n_iter=self.cfg.num_ISTA_iterations_eval,
                    tag=f'{self.itr:06d}')

            self.itr += 1

        # self.plot_single(n_iter)


    def plot_single(self, n_iter, tag='t'):
        inst = self.sample_single_inst(seed=0)

        x_cvxpy = self.solve_cvxpy(inst)
        cvxpy_obj = self.obj(inst, x_cvxpy).cpu()
        print(f'=== CVXPY, objective value: {cvxpy_obj:.2e}')
        print(to_np(x_cvxpy)[:10])

        d_ISTA = self.solve_ISTA(inst, n_iter=n_iter, track_iterates=True)
        x = d_ISTA["solution"]
        final_obj = d_ISTA["objs"][-1]
        print(f'\n=== ISTA, final objective value: {final_obj:.2e}')
        print(to_np(x)[:10])

        # d_ISTA_AA = self.solve_ISTA_AA_old(inst, n_iter=n_iter, track_iterates=True)
        d_ISTA_AA = self.solve_ISTA_AA(inst, n_iter=n_iter, track_iterates=True)
        x = d_ISTA_AA["solution"]
        final_obj = d_ISTA_AA["objs"][-1]
        print(f'\n=== ISTA+AA, final objective value: {final_obj:.2e}')
        print(to_np(x)[:10])

        d_ISTA_neural = self.solve_ISTA_neural(
            inst, n_iter=n_iter, track_iterates=True)
        x = d_ISTA_neural["solution"]
        final_obj = d_ISTA_neural["objs"][-1]
        print(f'\n=== ISTA+neural, final objective value: {final_obj:.2e}')
        print(to_np(x)[:10])

        nrow, ncol = 1, 2
        fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow))

        ax = axs[0]
        ax.plot(d_ISTA['rel_residual_norms'], label='ISTA')
        ax.plot(d_ISTA_AA['rel_residual_norms'], label='ISTA_AA')
        ax.plot(d_ISTA_neural['rel_residual_norms'], label='ISTA_neural')
        ax.set_yscale('log')
        ax.set_xlabel('Fixed-Point Iteration')
        ax.set_ylabel('Fixed-Point Residual')
        # ax.legend()

        ax = axs[1]
        # ax.plot(d_ISTA['objs'], label='ISTA')
        # ax.plot(d_ISTA_AA['objs'], label='ISTA_AA')
        # ax.plot(d_ISTA_neural['objs'], label='ISTA_neural')
        ax.plot((cvxpy_obj-d_ISTA['objs'].cpu())**2, label='ISTA')
        ax.plot((cvxpy_obj-d_ISTA_AA['objs'].cpu())**2, label='ISTA_AA')
        ax.plot((cvxpy_obj-d_ISTA_neural['objs'].cpu())**2, label='ISTA_neural')
        ax.set_yscale('log')
        ax.set_xlabel('Fixed-Point Iteration')
        ax.set_ylabel('Objective Distance')
        # ax.legend()

        fname = tag + '.png'
        fig.tight_layout()
        fig.savefig(fname, transparent=True)
        plt.close(fig)

    def plot_agg(self, n_iter, tag='t', n_sample=50):
        ista_norms, ista_aa_norms, neural_norms = [], [], []
        ista_objs, ista_aa_objs, neural_objs = [], [], []
        for seed in range(n_sample):
            inst = self.sample_single_inst(seed=seed)

            x_cvxpy = self.solve_cvxpy(inst)
            cvxpy_obj = self.obj(inst, x_cvxpy).cpu()

            d_ISTA = self.solve_ISTA(inst, n_iter=n_iter, track_iterates=True)
            d_ISTA_AA = self.solve_ISTA_AA(
                inst, n_iter=n_iter, track_iterates=True)
            d_ISTA_neural = self.solve_ISTA_neural(
                inst, n_iter=n_iter, track_iterates=True)

            ista_norms.append(d_ISTA['rel_residual_norms'])
            ista_aa_norms.append(d_ISTA_AA['rel_residual_norms'])
            neural_norms.append(d_ISTA_neural['rel_residual_norms'])

            ista_objs.append((cvxpy_obj-d_ISTA['objs'].cpu())**2)
            ista_aa_objs.append((cvxpy_obj-d_ISTA_AA['objs'].cpu())**2)
            neural_objs.append((cvxpy_obj-d_ISTA_neural['objs'].cpu())**2)

        ista_norms = torch.tensor(ista_norms)
        ista_aa_norms = torch.tensor(ista_aa_norms)
        neural_norms = torch.tensor(neural_norms)
        ista_objs = torch.stack(ista_objs)
        ista_aa_objs = torch.stack(ista_aa_objs)
        neural_objs = torch.stack(neural_objs)

        nrow, ncol = 1, 2
        fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow))

        ax = axs[0]
        mean, std = ista_norms.mean(dim=0), ista_norms.std(dim=0)
        xs = np.arange(len(mean))
        l, = ax.plot(xs, mean, label='ISTA')
        ax.fill_between(xs, mean-std, mean+std, color=l.get_color(), alpha=0.2)

        mean, std = ista_aa_norms.mean(dim=0), ista_aa_norms.std(dim=0)
        l, = ax.plot(xs, mean, label='ISTA+AA')
        ax.fill_between(xs, mean-std, mean+std, color=l.get_color(), alpha=0.2)

        mean, std = neural_norms.mean(dim=0), neural_norms.std(dim=0)
        l, = ax.plot(xs, mean, label='ISTA+Neural')
        ax.fill_between(xs, mean-std, mean+std, color=l.get_color(), alpha=0.2)

        ax.set_yscale('log')
        ax.set_xlabel('Fixed-Point Iteration')
        ax.set_ylabel('Fixed-Point Residual')

        ax = axs[1]
        min_obj_dist = np.inf
        mean, std = ista_objs.mean(dim=0), ista_objs.std(dim=0)
        min_obj_dist = min(min_obj_dist, mean.min())
        l, = ax.plot(xs, mean, label='ISTA')
        ax.fill_between(xs, mean-std, mean+std, color=l.get_color(), alpha=0.2)

        mean, std = ista_aa_objs.mean(dim=0), ista_aa_objs.std(dim=0)
        min_obj_dist = min(min_obj_dist, mean.min())
        l, = ax.plot(xs, mean, label='ISTA+AA')
        ax.fill_between(xs, mean-std, mean+std, color=l.get_color(), alpha=0.2)

        mean, std = neural_objs.mean(dim=0), neural_objs.std(dim=0)
        min_obj_dist = min(min_obj_dist, mean.min())
        l, = ax.plot(xs, mean, label='ISTA+Neural')
        ax.fill_between(xs, mean-std, mean+std, color=l.get_color(), alpha=0.2)
        # ax.set_ylim(min_obj_dist, 1e3)

        ax.set_yscale('log')
        ax.set_xlabel('Fixed-Point Iteration')
        ax.set_ylabel('Objective Distance')

        fname = 'agg_' + tag + '.png'
        fig.tight_layout()
        fig.savefig(fname, transparent=True)
        plt.close(fig)
        os.system(f"convert {fname} -trim {fname}")


    def init_logging(self):
        self.logf = open('log.csv', 'a')
        fieldnames = ['iter', 'loss']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            self.writer.writeheader()


from enr import Workspace as W # For saving/loading

@hydra.main(config_name='enr.yaml')
def main(cfg):
    fname = os.getcwd() + '/latest.pt'
    if os.path.exists(fname):
        print(f'Resuming fom {fname}')
        with open(fname, 'rb') as f:
            workspace = torch.load(f)
            workspace.init_logging()
    else:
        workspace = W(cfg)

    workspace.run()


if __name__ == '__main__':
    main()
