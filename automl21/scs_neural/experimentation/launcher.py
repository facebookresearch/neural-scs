#!/usr/bin/env python3

import csv
import os
import torch
import copy

from .. import problem
from .. import solver
from .metrics import RunningAverageMeter
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.itr = 0
        self.loss_meter = RunningAverageMeter(0.9)
        self.val_loss_meter = RunningAverageMeter(0.5)
        self.test_loss_meter = RunningAverageMeter(0.5)
        self.device = cfg.device

        self.scs_neural = solver.NeuralScsBatchedSolver(
            model_cfg=cfg.model,
            device=self.device,
            unscale_before_model=self.cfg.unscale_before_model,
            use_sparse=self.cfg.use_sparse_matrix,
            use_jitted_cones=self.cfg.use_jitted_cones,
            regularize=self.cfg.regularize,
            use_unscaled_loss=self.cfg.use_unscaled_loss,
            seed=self.cfg.train_seed
        )
        plt.style.use('bmh')

    def _create_dataset(self):
        if self.cfg.problem_name == 'lasso':
            self._create_dataset_lasso()
        else:
            raise RuntimeError("Unknown problem name")

    def _create_dataset_lasso(self):
        k, n = self.cfg.lasso_var_base, self.cfg.lasso_cons_base
        self.scs_problem = problem.Lasso(
            k, n_samples=self.cfg.num_train_instances, train_data_size=n
        )
        self.scs_test_problem = problem.Lasso(
            k, n_samples=self.cfg.num_test_instances, train_data_size=n
        )
        self.scs_validate_problem = problem.Lasso(
            k, n_samples=self.cfg.num_validate_instances, train_data_size=n
        )

    def _learn_batched(self):
        self.lowest_val_loss = -1
        if self.cfg.use_train_seed:
            torch.manual_seed(self.cfg.train_seed)
        rng_train_data = np.random.default_rng(self.cfg.train_data_seed)
        self.scs_neural.create_model(self.scs_problem)
        if self.cfg.log_tensorboard:
            self.sw = SummaryWriter(log_dir=self.cfg.tensorboard_dir)
        self.opt = torch.optim.Adam(
            self.scs_neural.accel.parameters(),
            lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
        if self.cfg.cosine_lr_decay:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, self.cfg.num_model_updates)
        self.multi_instance = self.scs_neural.scale_and_cache_all_instances(
            self.scs_problem.instances, use_scaling=self.cfg.scs.use_problem_scaling,
            scale=self.cfg.scs.scale, rho_x=self.cfg.scs.rho_x
        )
        with torch.no_grad():
            self.val_multi_instance = self.scs_neural.scale_and_cache_all_instances(
                self.scs_validate_problem.instances, use_scaling=self.cfg.scs.use_problem_scaling,
                scale=self.cfg.scs.scale, rho_x=self.cfg.scs.rho_x
            )
            self.test_multi_instance = self.scs_neural.scale_and_cache_all_instances(
                self.scs_test_problem.instances, use_scaling=self.cfg.scs.use_problem_scaling,
                scale=self.cfg.scs.scale, rho_x=self.cfg.scs.rho_x
            )
        self._reset_diffu_counts()
        while self.itr < self.cfg.num_model_updates:
            sampled_ids = rng_train_data.choice(len(self.scs_problem.instances),
                                                size=self.cfg.train_batch_size,
                                                replace=False)
            num_tries = 0
            while True:
                num_tries += 1
                curr_multi_instance = self.scs_neural.select_instances(
                    self.multi_instance, sampled_ids)
                soln_neural, metrics, diffu_counts, loss_available = self.scs_neural.solve(
                        curr_multi_instance,
                        max_iters=self.cfg.num_iterations_train,
                        alpha=self.cfg.scs.alpha
                )
                if loss_available:
                    break
                if num_tries > 10000:
                    raise RuntimeError("Unable to find feasible train samples")
            losses = [soln_neural[i]['loss'] for i in range(self.cfg.train_batch_size)]
            loss, index_nans = self._compute_loss(losses)
            self.loss_meter.update(loss.item())
            self._update_diffu_counts(diffu_counts)
            self.opt.zero_grad()
            loss.backward()
            if self.cfg.clip_gradients:
                torch.nn.utils.clip_grad_norm_(
                    self.scs_neural.accel.parameters(),
                    self.cfg.max_gradient)
            if self.itr % self.cfg.test_freq == 0:
                if len(index_nans) == 0:
                    if self.cfg.log_tensorboard and hasattr(self.scs_neural.accel, 'log'):
                        self.scs_neural.accel.log(self.sw, self.itr)
            self.opt.step()
            if self.cfg.cosine_lr_decay:
                self.scheduler.step()
            if self.itr % self.cfg.test_freq == 0:
                print("Loss: ", loss.item())
                self._plot_test_results(
                    n_iter=self.cfg.num_iterations_eval,
                    dataset_type='validate'
                )
                test_results = self._plot_test_results(
                    n_iter=self.cfg.num_iterations_eval, tag=f'{self.itr:06d}',
                    dir_tag=f'{self.itr // 1000:03d}'
                )
                train_results = self._plot_train_results(
                    n_iter=self.cfg.num_iterations_eval, tag=f'{self.itr:06d}',
                    dir_tag=f'{self.itr // 1000:03d}'
                )
                if len(test_results) > 0 and len(train_results) > 0:
                    self.plot_aggregate_results(
                        test_results, train_results,
                        tag=f'{self.itr:06d}',
                        dir_tag=f'{self.itr // 1000:03d}'
                    )
                self._reset_diffu_counts()
            if self.itr % self.cfg.save_freq == 0:
                torch.save(self, 'latest.pt')
                if self.val_loss_meter.avg < self.lowest_val_loss or \
                   self.lowest_val_loss == -1:
                    torch.save(self, 'best_model.pt')
                    self.lowest_val_loss = self.val_loss_meter.avg
            self.itr += 1

    def _reset_diffu_counts(self):
        self.diff_u_sum = 0.
        self.diff_u_sum_sq = 0.
        self.count_sum = 0.

    def _update_diffu_counts(self, diffu_counts):
        diff_u_per_iter, diff_u_sq_per_iter, count_diff_u_per_iter = diffu_counts
        diff_u_sq_per_iter[torch.isnan(diff_u_per_iter)] = 0.
        count_diff_u_per_iter[torch.isnan(diff_u_per_iter)] = 0.
        diff_u_per_iter[torch.isnan(diff_u_per_iter)] = 0.
        self.diff_u_sum += diff_u_per_iter
        self.diff_u_sum_sq += diff_u_sq_per_iter
        self.count_sum += count_diff_u_per_iter

    def _compute_loss(self, losses):
        loss_nan_index, losses2 = [], []
        for i, curr_loss in enumerate(losses):
            if torch.isnan(curr_loss):
                loss_nan_index.append(i)
            else:
                losses2.append(curr_loss)
        if len(losses2) == 0:
            print(losses)
            return -1, loss_nan_index
        if len(loss_nan_index) > 0:
            print("Discarded losses for instances: ", len(loss_nan_index))
        loss = sum(losses2) / len(losses2)
        return loss, loss_nan_index

    def _plot_metrics(self, data, nrow, ncol, metrics, metric_names, tag, std=None):
        fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow))
        for i in range(len(metrics)):
            if nrow > 1 and ncol > 1:
                ax = axs[int(i / 2)][i % 2]
            elif nrow > 1 or ncol > 1:
                ax = axs[i]
            else:
                ax = axs
            metric = metrics[i]
            metric_name = metric_names[i]
            if metric == 'loss':
                x = [i * self.cfg.test_freq for i in range(len(data['scs_neural'][metric]['train']))]
                ax.plot(x, data['scs_neural'][metric]['train'], label='training loss')
                ax.plot(x, data['scs_neural'][metric]['val'], label='validation loss')
                ax.plot(x, data['scs_neural'][metric]['test'], label='test loss')
                ax.set_yscale('log')
                ax.set_xlabel('Training Iteration')
            elif metric == 'all_u_diff_train':
                x = [i for i in range(len(data['scs_neural'][metric]))]
                delta_z = std['scs_neural'][metric]
                lower = [0 for i in range(len(data['scs_neural'][metric]))]
                upper = delta_z
                ax.errorbar(x, data['scs_neural'][metric], (lower, upper), ecolor='red', label='scs_neural')
                ax.set_yscale('log')
                ax.set_xlim((0, len(x)))
            elif std is not None:
                for m, name in enumerate(['scs_neural']):
                    x = [i for i in range(len(data[name][metric]))]
                    delta_z = std[name][metric]
                    lower = [0 for i in range(len(data[name][metric]))]
                    upper = delta_z
                    ax.errorbar(x, data[name][metric], (lower, upper), label=name)
                ax.set_xlabel('Fixed-Point Iteration')
                ax.set_yscale('log')
            else:
                ax.plot(data['scs_neural'][metric], label='scs_neural')
                ax.set_xlabel('Fixed-Point Iteration')
                ax.set_yscale('log')
                if metric in ['p_res', 'd_res']:
                    limits = self._find_y_limits([data['scs_neural'][metric]])
                    ax.set_ylim(limits)
            ax.set_ylabel(metric_name)
            ax.legend()

        fig.tight_layout()
        fname = tag + '.png'
        fig.savefig(fname)
        plt.close(fig)

    def _find_y_limits(self, data, limit_u=1e4):
        x_max = 0
        x_min = 1e4
        for i in range(len(data)):
            x = [x for x in data[i] if x < limit_u]
            x_max = max(x_max, max(x))
            x_min = min(x_min, min(x))
        return (x_min * 0.8, x_max * 1.2)

    def _plot_test_results(self, dataset_type='test', n_iter=10, tag='t', dir_tag=None):
        if dataset_type == 'validate':
            problems, multi_instance = self.scs_validate_problem, self.val_multi_instance 
            batch_size, graph_batch_size = self.cfg.validate_batch_size, self.cfg.validate_graph_batch_size
        else:
            problems, multi_instance = self.scs_test_problem, self.test_multi_instance
            batch_size, graph_batch_size = self.cfg.test_batch_size, self.cfg.test_graph_batch_size
        with torch.no_grad():
            if multi_instance.num_instances == batch_size:
                soln_neural, scs_neural_metrics = self.scs_neural.solve(
                    multi_instance, max_iters=n_iter, track_metrics=True, train=False)
            else:
                all_soln_neural, all_neural_metrics = [], []
                for i in range(0, multi_instance.num_instances, batch_size):
                    max_instance_id = min((i + batch_size), multi_instance.num_instances)
                    curr_test = self.scs_neural.select_instances(
                        multi_instance, 
                        [x for x in range(i, max_instance_id)])
                    soln_neural, scs_neural_metrics = self.scs_neural.solve(
                        curr_test, max_iters=n_iter, track_metrics=True, train=False)
                    all_soln_neural = all_soln_neural + soln_neural
                    all_neural_metrics = all_neural_metrics + scs_neural_metrics
                soln_neural, scs_neural_metrics = all_soln_neural, all_neural_metrics
        
        losses = [soln_neural[i]['loss'] for i in range(len(soln_neural))]
        loss, index_nans = self._compute_loss(losses)

        if dataset_type == 'validate':
            if loss > 0:
                self.val_loss_meter.update(loss.item())
            return

        if dataset_type == 'test':
            if loss > 0:
                self.test_loss_meter.update(loss.item())

        self.writer.writerow({
            'iter': self.itr,
            'train_loss': "%.6e" % (self.loss_meter.avg),
            'val_loss': "%.6e" %(self.val_loss_meter.avg),
            'test_loss': "%.6e" %(self.test_loss_meter.avg)
        })
        self.logf.flush()
        if self.cfg.log_tensorboard:
            self.sw.add_scalars("Loss", {"train": self.loss_meter.avg,
                                         "validate": self.val_loss_meter.avg,
                                         "test": self.test_loss_meter.avg},
                                self.itr)
        if loss == -1:
            return []

        if dir_tag is None:
            dir_tag = ""
        upd_dir_tag = "test/" + dir_tag

        x = [x for x in range(len(problems.instances))]
        if len(index_nans) > 0:
            x = [i for i in x if i not in index_nans]
        
        sampled_ids = np.random.choice(
            x,
            size=graph_batch_size,
            replace=False
        )

        agg_scs_neural, conf_scs_neural = self._extract_aggregate_metrics(
            scs_neural_metrics, soln_type='neural', index_nans=index_nans
        )
        self._plot_solution_results(sampled_ids, scs_neural_metrics, tag=tag,
                                    dir_tag=upd_dir_tag, title_stub='Test')
        return agg_scs_neural, conf_scs_neural

    def _plot_solution_results(self, sampled_ids, scs_neural_metrics, tag='t', 
                               dir_tag=None, title_stub=''):
        os.makedirs('aggregates', exist_ok=True)
        os.makedirs('samples', exist_ok=True)

        if dir_tag is not None:
            os.makedirs('samples/' + dir_tag, exist_ok=True)
            tag = dir_tag + '/' + tag

        data = {}
        for curr_id in sampled_ids:
            print(f'\n=== Instance sample: {curr_id:03d}')
            data["scs_neural"] = self._extract_individual_metrics(
                scs_neural_metrics[curr_id], soln_type='neural'
            )
            p_obj, d_obj = data["scs_neural"]["p_obj"][-1], data["scs_neural"]["d_obj"][-1]
            print(f'=== SCS+neural, final objective value: {p_obj:.5e} {d_obj:.5e}')

            nrow, ncol = 3, 2
            metrics = ['u_diff', 'p_res', 'd_res', 'p_obj', 'd_obj', 'tau']
            metric_names = ['Fixed-Point Residual', 'Primal Residual', 'Dual Residual',
                            'Primal objective', 'Dual objective', 'tau']
            upd_tag = 'samples/' + tag + '_' + f'{curr_id:03d}' 
            self._plot_metrics(data, nrow, ncol, metrics, metric_names, upd_tag)


    def plot_aggregate_results(self, test_results, train_results,
                               tag='t', dir_tag=None):
        if dir_tag is None:
            dir_tag = ""
        upd_dir_tag = 'aggregates/' + dir_tag + '/'
        if dir_tag is not None:
            os.makedirs('aggregates/' + dir_tag, exist_ok=True)
        
        nrow, ncol = 3, 2
        metrics = ['u_diff_train', 'u_diff_test', 'u_diff_orig_train', 'u_diff_orig_test', 
                   'all_u_diff_train', 'loss']
        metric_names = ['Normalized |u - u_prev| (Train)', 'Normalized |u - u_prev| (Test)',
                        '|u - u_prev| (Train)', '|u - u_prev| (Test)', 
                        'Unscaled |u - u_prev| Train', 'Loss']

        upd_tag = upd_dir_tag + tag + '_agg'

        agg, conf = [{}, {}, {}], [{}, {}, {}]

        all_metrics = ['u_diff', 'u_diff_orig']
        for metric in all_metrics:
            results = [train_results, test_results]
            for i, name in enumerate(['train', 'test']):
                metric_upd = metric + '_' + name
                curr_results = results[i]
                for j in range(1):
                    agg[j][metric_upd] = curr_results[2*j][metric]
                    conf[j][metric_upd] = curr_results[2*j + 1][metric]
        metric = 'loss'
        for j in range(1):
            agg[j][metric] = test_results[2*j][metric]

        data = {"scs_neural": agg[0]}
        std = {"scs_neural": conf[0]}
        
        data['scs_neural']['all_u_diff_train'] = (self.diff_u_sum / self.count_sum).detach().numpy()
        var = (self.diff_u_sum_sq / self.count_sum - (self.diff_u_sum / self.count_sum).pow(2))
        std['scs_neural']['all_u_diff_train'] = var.sqrt().detach().numpy()

        self._plot_metrics(data, nrow, ncol, metrics, metric_names, upd_tag, std)

    def _plot_train_results(self, n_iter=10, tag='t', dir_tag=None, baseline=False):
        if not hasattr(self, 'longitudinal_samples'):
            self.longitudinal_samples = np.random.randint(
                0, len(self.scs_problem.instances),
                size=self.cfg.num_test_instances
            )        
        multi_instance = self.scs_neural.select_instances(self.multi_instance,
                                                          self.longitudinal_samples)
        sampled_instances = [self.scs_problem.instances[curr_id] for curr_id in self.longitudinal_samples]
        batch_size = self.cfg.train_batch_size

        if multi_instance.num_instances == batch_size:
            soln_neural, scs_neural_metrics = self.scs_neural.solve(
                multi_instance, max_iters=n_iter, track_metrics=True, train=False)
        else:
            all_soln_neural, all_neural_metrics = [], []
            for i in range(0, multi_instance.num_instances, batch_size):
                max_instance_id = min((i + batch_size), multi_instance.num_instances)
                curr_test = self.scs_neural.select_instances(
                    multi_instance, 
                    [x for x in range(i, max_instance_id)])
                soln_neural, scs_neural_metrics = self.scs_neural.solve(
                    curr_test, max_iters=n_iter, track_metrics=True, train=False)
                all_soln_neural = all_soln_neural + soln_neural
                all_neural_metrics = all_neural_metrics + scs_neural_metrics
            soln_neural, scs_neural_metrics = all_soln_neural, all_neural_metrics

        losses = [soln_neural[i]['loss'] for i in range(len(soln_neural))]
        loss, index_nans = self._compute_loss(losses)
        if loss == -1:
            return []

        if dir_tag is None:
            dir_tag = ""
        upd_dir_tag = "train/" + dir_tag

        x = [i for i in range(len(self.longitudinal_samples))]
        if len(index_nans) > 0:
            x = [i for i in x if i not in index_nans]

        sampled_ids = np.random.choice(
            x, size=self.cfg.train_graph_batch_size, replace=False
        )

        agg_scs_neural, conf_scs_neural = self._extract_aggregate_metrics(
            scs_neural_metrics, soln_type='neural', index_nans=index_nans)
    
        self._plot_solution_results(sampled_ids, scs_neural_metrics, tag=tag, dir_tag=upd_dir_tag, title_stub='Train')
        return agg_scs_neural, conf_scs_neural

    def _extract_individual_metrics(self, soln_metrics, soln_type='original'):
        result_dict = {}
        if soln_type == 'neural':
            residuals, objectives = soln_metrics[0], soln_metrics[1]
            result_dict['p_res'] = np.array([x[0] for x in residuals])
            result_dict['d_res'] = np.array([x[1] for x in residuals])
            result_dict['p_obj'] = np.array([x[0] for x in objectives])
            result_dict['d_obj'] = np.array([x[1] for x in objectives])
            result_dict['tau'] = np.array(soln_metrics[2])
            result_dict['u_diff'] = np.array(soln_metrics[-1])
        else:
            raise RuntimeError("Unknown solution type")

        pobj = result_dict['p_obj']
        result_dict['p_obj'][pobj > 1e5] = np.nan
        dobj = result_dict['d_obj']
        result_dict['d_obj'][dobj > 1e5] = np.nan

        # normalize diff_u
        orig_iterate_diff = result_dict['u_diff'][0]
        if hasattr(self.cfg.model, 'learn_init_iterate') and self.cfg.model.learn_init_iterate and soln_type == 'neural':
            slice_start = 1
        else:
            slice_start = 0
        data = [x / orig_iterate_diff for x in result_dict['u_diff'][slice_start:]]
        result_dict['u_diff'] = data
        return result_dict

    def _extract_aggregate_metrics(self, soln_metrics, soln_type='original', index_nans=[]):
        all_data, all_metrics = [], []
        result_dict, conf_dict = {}, {}
        all_data_orig, all_metrics_orig = [], []
        max_data_len = 0
        for i in range(len(soln_metrics)):
            if i in index_nans:
                continue
            if soln_type == 'neural':
                u_diff_instance = soln_metrics[i][-1]
                u_diff_instance = [x.cpu() for x in u_diff_instance]
            else:
                raise RuntimeError("Unknown solution type")
            orig_iterate_diff = u_diff_instance[0]
            if hasattr(self.cfg.model, 'learn_init_iterate') and self.cfg.model.learn_init_iterate and soln_type == 'neural':
                slice_start = 1
            else:
                slice_start = 0
            data = [x / orig_iterate_diff for x in u_diff_instance[slice_start:]]
            all_data.append(data)
            all_data_orig.append(u_diff_instance)
            max_data_len = max(len(data), max_data_len)

        for i in range(len(all_data)):
            curr_metrics = np.array(all_data[i])
            diff = max_data_len - len(curr_metrics)
            if max_data_len > 0:
                upd_curr_metrics = np.pad(curr_metrics, (0, diff), 'constant',
                                          constant_values=(curr_metrics[-1]))
            else:
                upd_curr_metrics = curr_metrics
            all_metrics.append(upd_curr_metrics)

        for i in range(len(all_data_orig)):
            curr_metrics = np.array(all_data_orig[i])
            diff = max_data_len + 1 - len(curr_metrics)
            if max_data_len > 0:
                upd_curr_metrics = np.pad(curr_metrics, (0, diff), 'constant',
                                          constant_values=(curr_metrics[-1]))
            else:
                upd_curr_metrics = curr_metrics
            all_metrics_orig.append(upd_curr_metrics)

        final_metrics = np.stack(all_metrics)
        result_dict['u_diff'] = final_metrics.mean(axis=0)
        std_dev = final_metrics.std(axis=0)
        conf_dict['u_diff'] = std_dev

        final_metrics_orig = np.stack(all_metrics_orig)
        result_dict['u_diff_orig'] = final_metrics_orig.mean(axis=0)
        std_dev = final_metrics_orig.std(axis=0)
        conf_dict['u_diff_orig'] = std_dev

        # add data from log file:
        loss_data = pd.read_csv('log.csv', delimiter=',', header=0)
        train_loss = loss_data['train_loss'].to_numpy()
        val_loss = loss_data['val_loss'].to_numpy()
        test_loss = loss_data['test_loss'].to_numpy()
        result_dict["loss"] = {}
        result_dict["loss"]["train"] = train_loss
        result_dict["loss"]["val"] = val_loss
        result_dict["loss"]["test"] = test_loss

        return result_dict, conf_dict

    def _init_logging(self):
        self.logf = open('log.csv', 'a')
        fieldnames = ['iter', 'train_loss', 'val_loss', 'test_loss']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            self.writer.writeheader()

    def __getstate__(self):
        # do not store problem data
        multi_instance = self.__dict__.pop('multi_instance', None)
        val_multi_instance = self.__dict__.pop('val_multi_instance', None)
        test_multi_instance = self.__dict__.pop('test_multi_instance', None)
        scs_problem = self.__dict__.pop('scs_problem', None)
        scs_validate_problem = self.__dict__.pop('scs_validate_problem', None)
        scs_test_problem = self.__dict__.pop('scs_test_problem', None)
        sw = self.__dict__.pop('sw', None)
        logf, writer = self.__dict__.pop('logf'), self.__dict__.pop('writer')

        # move model to cpu
        self.scs_neural.accel = self.scs_neural.accel.to('cpu')

        state = copy.deepcopy(self.__dict__)

        self.scs_neural.accel = self.scs_neural.accel.to(self.cfg.model.device)
        self.multi_instance = multi_instance
        self.val_multi_instance = val_multi_instance
        self.test_multi_instance = test_multi_instance
        self.scs_problem = scs_problem
        self.scs_validate_problem = scs_validate_problem
        self.scs_test_problem = scs_test_problem
        self.sw = sw
        self.logf, self.writer = logf, writer
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def run(self):
        self._init_logging()
        print("Creating dataset...")
        self._create_dataset()
        print("Dataset created.")
        self._learn_batched()
        self._plot_test_results(
            dataset_type='validate',
            n_iter=self.cfg.num_iterations_eval, 
            tag='final', dir_tag='final'
        )
        test_results = self._plot_test_results(
            n_iter=self.cfg.num_iterations_eval, 
            tag='final', dir_tag='final',
        )
        train_results = self._plot_train_results(
            n_iter=self.cfg.num_iterations_eval, 
            tag='final', dir_tag='final',
        )
        self.plot_aggregate_results(
            test_results, train_results,
            tag='final',
            dir_tag='final',
        )
        self._reset_diffu_counts()
        torch.save(self, 'latest.pt')
        if self.val_loss_meter.avg < self.lowest_val_loss:
            torch.save(self, 'best_model.pt')
        return self.lowest_val_loss
