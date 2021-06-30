# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python3

import torch
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy as np

from ..utils import ConeUtils
from hydra.utils import instantiate

from .solver import Solver
from .cone_projection import ConeProjection
from .neural_lu_solve import NeuralLuSolve, NeuralSparseLuSolve
from accel.neural_rec import NeuralRec
from ..problem import ScsMultiInstance


class NeuralScsBatchedSolver(Solver):
    def __init__(self,
                 model_cfg, unscale_before_model=False,
                 device='cpu',
                 use_sparse=True, use_jitted_cones=False,
                 regularize=0.0,
                 use_unscaled_loss=True,
                 seed=0,
                 **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.unscale_before_model = unscale_before_model
        self.device = device
        self.use_sparse = use_sparse
        self.use_jitted_cones = use_jitted_cones
        self.regularize = regularize
        self.use_unscaled_loss = use_unscaled_loss
        self.seed = seed

    def create_model(self, problems, **kwargs):
        m, n = problems.get_sizes()
        scale_count = 0 if self.unscale_before_model else 1
        iterate_size = m + n + scale_count
        context_size = m * n + m + n
        
        self.accel = instantiate(
            self.model_cfg,
            iterate_size=iterate_size,
            context_size=context_size,
        ).to(device=self.model_cfg.device)
        ConeProjection.initialize_cones(problems.get_cone(), m, n,
                                        use_jit=self.use_jitted_cones)

    def solve(self, multi_instance, train=True, max_iters=5000, use_scaling=True, scale=1,
              rho_x=1e-3, alpha=1.5, track_metrics=False, **kwargs):
        # create a set of sets for accessing all past iterates
        diffs_u, objectives, all_residuals, losses = [], [], [], []
        seq_tau, train_diff_u = [], []

        QplusI_lu, QplusI_t_lu = self._obtain_QplusI_matrices(multi_instance, rho_x)
        m, n, num_instances = multi_instance.get_sizes()
        total_size = m + n + 1
        u, v = self._initialize_iterates(m, n, num_instances)

        context = None
        init_diff_u, init_fp_u, init_scaled_u = self._compute_init_diff_u(
            u, v, total_size, multi_instance, rho_x, alpha
        )

        if isinstance(self.accel, NeuralRec):
            if self.model_cfg.learn_init_iterate:
                diffs_u.append(init_diff_u.norm(dim=1))

            if self.model_cfg.learn_init_iterate or \
                    self.model_cfg.learn_init_hidden:
                context = self._construct_context(multi_instance)
                context = context.to(self.model_cfg.device)

        u, tau, scaled_u = self._unscale_before_model(u)
        u = u.to(self.model_cfg.device)
        u, hidden = self.accel.init_instance(
            init_x=u, context=context)
        u = u.to(self.device)
        u_orig = self._rescale_after_model(u, tau)

        for j in range(max_iters):
            if j > 0:
                u_upd, tau, scaled_u = self._unscale_before_model(u)
                fp_u_upd, fp_tau, scaled_fp_u = self._unscale_before_model(fp_u)
                if scaled_u and scaled_fp_u:
                    u_upd, fp_u_upd = u_upd.to(self.model_cfg.device), fp_u_upd.to(self.model_cfg.device)
                    u_upd, _, hidden = self.accel.update(
                        fx=fp_u_upd, x=u_upd, hidden=hidden)
                    u_upd = u_upd.to(self.device)
                else:
                    u_upd, tau = fp_u_upd, fp_tau
                u_orig = self._rescale_after_model(u_upd, tau)

            u, v = self._scale_iterates(u_orig, v, total_size)
            fp_u, fp_v = self._fixed_point_iteration(
                QplusI_lu, QplusI_t_lu, u, v, multi_instance, rho_x, alpha
            )
            v = fp_v

            diff_u = fp_u - u
            with torch.no_grad():
                curr_train_diff_u = self._compute_scaled_loss(u_orig, fp_u)
                train_diff_u.append(curr_train_diff_u)
            if j > 0:
                curr_loss = self._compute_scaled_loss(u_orig, fp_u)
                if curr_loss is not None:
                    losses.append(curr_loss)
            if track_metrics:
                with torch.no_grad():
                    res_p, res_d = self._compute_residuals(fp_u, fp_v,
                                                           multi_instance)
                    all_residuals.append([res_p.norm(dim=1), res_d.norm(dim=1)])
                    diffs_u.append(diff_u.norm(dim=1))
                    seq_tau.append(fp_u[:, -1])
                    objectives.append(self._get_objectives(fp_u, fp_v,
                                                           multi_instance))

        # check if the solution is feasible
        all_tau = fp_u[:, -1]
        # if any solution is infeasible, zero it out from the loss
        if (all_tau <= 0).any():
            curr_loss = self._compute_scaled_loss(u_orig, fp_u, include_bad_tau=True)
            if curr_loss is not None:
                losses.append(curr_loss)
        if (all_tau <= 0).all() or len(losses) == 0:
            return [], [], [], False

        # convert iterates to solution, and track all solutions
        if train and self.regularize > 0.:
            res_p, res_d = self._compute_residuals_sparse_for_backprop(fp_u, fp_v, multi_instance)
        else:
            if not track_metrics:
                with torch.no_grad():
                    res_p, res_d = self._compute_residuals(fp_u, fp_v, multi_instance)
        soln, diffu_counts = self._extract_solution(
            fp_u, fp_v, multi_instance, (res_p, res_d), losses, train_diff_u
        )
        metrics = {}
        if track_metrics:
            metrics = {"residuals": all_residuals, "objectives": objectives,
                       "diffs_u": diffs_u, "all_tau": seq_tau}
        # do this to be consistent with SCS
        soln, metrics = self._convert_to_sequential_list(soln, metrics, num_instances)
        if train:
            return soln, metrics, diffu_counts, True
        else:
            return soln, metrics

    def _construct_context(self, multi_instance):
        if self.use_sparse:
            A_reshape_arr = [multi_instance.A[i].reshape(1, -1) for i in range(multi_instance.num_instances)]
            A_reshaped = sp.vstack(A_reshape_arr)
        else:
            A_reshaped = multi_instance.A.reshape(multi_instance.num_instances, -1)
        b_reshaped = multi_instance.b.reshape(multi_instance.num_instances, -1)
        c_reshaped = multi_instance.c.reshape(multi_instance.num_instances, -1)

        if self.use_sparse:
            A_reshaped = torch.from_numpy(A_reshaped.toarray())
        context = torch.cat([A_reshaped, b_reshaped, c_reshaped], dim=1)
        return context

    def _initialize_iterates(self, m, n, num_instances):
        """Initialize all iterates identically to SCS"""
        total_size = m + n + 1
        u = torch.zeros((num_instances, total_size), dtype=torch.double)
        v = torch.zeros((num_instances, total_size), dtype=torch.double)
        u[:, -1] = np.sqrt(total_size)  
        v[:, -1] = np.sqrt(total_size)
        u, v = u.to(self.device), v.to(self.device)
        return u, v

    def _compute_init_diff_u(self, u, v, total_size, multi_instance, rho_x, alpha):
        QplusI_lu, QplusI_t_lu = self._obtain_QplusI_matrices(multi_instance, rho_x)
        new_u, new_v = self._scale_iterates(u, v, total_size)
        fp_u, fp_v = self._fixed_point_iteration(
                QplusI_lu, QplusI_t_lu, new_u, new_v, multi_instance, rho_x, alpha
        )
        diff_u = fp_u - u
        return diff_u, fp_u, new_u

    def _scale_iterates(self, last_u, last_v, total_size):
        total_size_t = torch.tensor(total_size).double()
        iterate_norm = 10
        total_norm = (last_u.norm(dim=1).pow(2) + last_v.norm(dim=1).pow(2)).sqrt()
        total_norm = total_norm.unsqueeze(1)
        last_u = last_u * torch.sqrt(total_size_t) * iterate_norm / total_norm
        last_v = last_v * torch.sqrt(total_size_t) * iterate_norm / total_norm
        return last_u, last_v

    def _compute_scaled_loss(self, u, fp_u, include_bad_tau=False):
        if self.use_unscaled_loss:
            u, _, scaled_u = self._remove_tau_scaling_from_iterate(u, include_bad_tau)
            fp_u, _, scaled_fp_u = self._remove_tau_scaling_from_iterate(fp_u, include_bad_tau)
            if not scaled_u or not scaled_fp_u:
                return None
        else:
            tau = u[:, -1]
            fp_tau = u[:, -1]
            if not include_bad_tau:
                if (tau <= 0).any() or (fp_tau <= 0).any():
                    return None
        diff_u = (fp_u - u).norm(dim=1)
        return diff_u

    def _apply_model_to_instance(self, u, v, diff_u, h, c):
        """Apply the learned model to get the iterates"""
        g = diff_u  
        x, h, c = self.model(u, v, g, h, c)
        size = int(x.size(-1)/2)
        arr = torch.split(x, size, dim=-1)
        model_u, model_v = arr
        return model_u, model_v, h, c

    def _obtain_QplusI_matrices(self, multi_instance, rho_x):
        """Compute and cache the QplusI matrices if necessary"""
        if hasattr(multi_instance, "QplusI_lu"):
            QplusI_lu = multi_instance.QplusI_lu
        else:
            assert self.use_sparse is False, 'Not implemented for sparse yet'
            QplusI = self._compute_QplusI(multi_instance, rho_x)
            QplusI_lu = torch.lu(QplusI)
            multi_instance.QplusI_lu = QplusI_lu

        if hasattr(multi_instance, "QplusI_t_lu"):
            QplusI_t_lu = multi_instance.QplusI_t_lu
        else:
            assert self.use_sparse is False, 'Not implemented for sparse yet'
            QplusI_t_lu = torch.lu(QplusI.transpose(1, 2))
            multi_instance.QplusI_t_lu = QplusI_t_lu
        return QplusI_lu, QplusI_t_lu

    def _apply_cone_projection(self, z, cones, m, n):
        upd_u = ConeProjection.apply(z, cones, m, n)
        return upd_u

    def _remove_tau_scaling_from_iterate(self, fp_u, include_bad_tau=False):
        tau = fp_u[:, -1].unsqueeze(1)
        unscaled_u = fp_u[:, :-1]
        scaling_used = False
        if (tau > 0).all():
            unscaled_u = unscaled_u / tau
            scaling_used = True
        elif include_bad_tau and (tau > 0).any():
            tau_copy = (tau > 0) * tau + (tau <= 0) * 1
            unscaled_u = unscaled_u / tau_copy
            unscaled_u = unscaled_u * (tau > 0) + 0 * (tau <= 0)
            scaling_used = True
        return unscaled_u, tau, scaling_used

    def _unscale_before_model(self, fp_u):
        """Scale the u by tau"""
        tau = fp_u[:, -1].unsqueeze(1)
        if self.unscale_before_model:
            return self._remove_tau_scaling_from_iterate(fp_u)
        else:
            return fp_u, tau, True

    def _rescale_after_model(self, model_u, tau):
        """Scale back by tau"""
        if self.unscale_before_model:
            if (tau > 0).all():
                updated_u = model_u * tau
            else:
                updated_u = model_u
            rescaled_u = torch.cat([updated_u, tau], dim=1)
            return rescaled_u
        else:
            return model_u

    def _extract_solution(self, u, v, multi_instance, residuals, losses, train_diff_u):
        """Convert the iterates to (rescaled) solution objects"""
        x, y, s, prim_obj, dual_obj = self._get_objectives(
            u, v, multi_instance, include_solution=True)
        prim_res, dual_res = residuals
        soln = {"x": x, "y": y, "s": s, "pobj": prim_obj,
                "dobj": dual_obj,
                "prim_res": prim_res.norm(dim=1),
                "dual_res": dual_res.norm(dim=1)}
        soln['loss'] = self._compute_overall_aggregate_loss(u, losses, residuals)
        diffu_counts = self._compute_diffu_counts(u, train_diff_u)
        return soln, diffu_counts

    def _compute_overall_aggregate_loss(self, u, losses, residuals):
        all_tau = u[:, -1]
        # stack the losses of the individual iterations: 
        # now the iterations are in dim=0, examples in dim=1
        all_losses_tensor = torch.stack(losses)
        # take the mean across all iterations for the same example
        loss = all_losses_tensor.mean(dim=0)
        # zero out all examples where the tau <= 0
        loss2 = loss * (all_tau > 0) + 0. * (all_tau <= 0)
        res_norm = torch.cat(residuals, dim=1).norm(dim=1)
        res_norm2 = torch.isnan(res_norm) * 0. + (~torch.isnan(res_norm)) * res_norm
        val = (1 - self.regularize) * loss2 + self.regularize * res_norm2
        return val

    def _compute_diffu_counts(self, u, train_diff_u):
        # stack the losses of individual iterations again: 
        losses_with_zeroes = [i if i is not None else torch.zeros_like(u.norm(dim=1)) for i in train_diff_u]
        count_per_iter = torch.tensor([u.size(0) if i is not None else 0 for i in train_diff_u], device=u.device)
        all_losses_tensor = torch.stack(losses_with_zeroes)
        # compute sum and sum of sq across examples.
        loss_per_iter = all_losses_tensor.sum(dim=1)
        loss_sq_per_iter = (all_losses_tensor.pow(2)).sum(dim=1)
        diffu_counts = [loss_per_iter, loss_sq_per_iter, count_per_iter]
        return diffu_counts

    def _fixed_point_iteration(self, QplusI_lu, QplusI_t_lu, u, v,
                               multi_instance, rho_x, alpha):
        """ SCS algorithm fixed point iteration loop """
        m, n, num_instances = multi_instance.get_sizes()
        u_tilde = self._solve_linear_system(QplusI_lu, QplusI_t_lu, u, v, rho_x, n)
        u_tilde = alpha * u_tilde + (1 - alpha) * u  # apply over-relaxation
        fp_u = self._apply_cone_projection(
            u_tilde - v, multi_instance.all_cones, m, n)
        fp_v = v - u_tilde + fp_u
        return fp_u, fp_v

    def _solve_linear_system(self, QplusI_lu, QplusI_t_lu, u, v, rho_x, n):
        """Solve the linear system as Step 1 of SCS"""
        uv_sum = (u + v)
        uv_sum[:, :n] *= rho_x
        if self.use_sparse:
            u_tilde = NeuralSparseLuSolve.apply(uv_sum, QplusI_lu, QplusI_t_lu)
        else:
            u_tilde = NeuralLuSolve.apply(uv_sum.unsqueeze(2), QplusI_lu, QplusI_t_lu)
            u_tilde = u_tilde.squeeze()
        return u_tilde

    def scale_and_cache_all_instances(self, instances, use_scaling=True, scale=1, rho_x=1e-3):
        if self.use_sparse:
            return self.scale_and_cache_all_instances_sparse(instances, use_scaling, scale, rho_x)
        else:
            return self.scale_and_cache_all_instances_dense(instances, use_scaling, scale, rho_x)

    def scale_and_cache_all_instances_dense(self, instances, use_scaling=True, scale=1, rho_x=1e-3):
        """Scales, computes and caches LU factor of Q+I for multi-instance"""
        multi_instance = self._create_multi_instance(instances)
        if use_scaling:
            upd_multi_instance = self._scale_instances_dense(multi_instance, scale)
        else:
            upd_multi_instance = multi_instance
        QplusI = self._compute_QplusI(upd_multi_instance, rho_x)
        QplusI_lu = torch.lu(QplusI)
        upd_multi_instance.QplusI_lu = QplusI_lu
        QplusI_t_lu = torch.lu(QplusI.transpose(1, 2))
        upd_multi_instance.QplusI_t_lu = QplusI_t_lu
        return upd_multi_instance

    def _create_multi_instance(self, instances):
        """Create an ScsMultiInstance object from list of instances"""
        all_cones = []
        for i in range(len(instances)):
            all_cones.append(instances[i].cones)
        A = np.stack([curr_instance.data["A"].toarray() for curr_instance in instances])
        b = np.stack([curr_instance.data["b"] for curr_instance in instances])
        c = np.stack([curr_instance.data["c"] for curr_instance in instances])
        multi_instance = ScsMultiInstance(A, b, c, all_cones, device=self.device)
        return multi_instance

    def _scale_instances_dense(self, multi_instance, scale):
        """Create the scaled instance using the original instance"""
        all_boundaries = []
        for i in range(multi_instance.num_instances):
            boundaries = ConeUtils.get_cone_boundaries(multi_instance.all_cones[i])
            all_boundaries.append(boundaries)
        boundaries_matrix = torch.tensor(all_boundaries, device=self.device)
        A, b, c = multi_instance.A, multi_instance.b, multi_instance.c
        A_upd, D, E, mean_row_norm, mean_col_norm = self._normalize_A_dense(
                                                    A, boundaries_matrix, scale
                                                )
        b_upd, c_upd, sigma, rho = self._normalize_b_c(
                                b, c, D, E, mean_row_norm, mean_col_norm, scale
                            )
        multi_instance = ScsMultiInstance(A_upd, b_upd, c_upd,
                                          multi_instance.all_cones,
                                          (D, E, sigma, rho, b, c),
                                          device=self.device)
        return multi_instance

    def _normalize_A_dense(self, A, boundaries, scale):
        """
           Normalize the A matrix. This code comes from: 
           https://github.com/bodono/scs-python/blob/master/test/test_scs_python_linsys.py
        """
        num_instances = len(A)
        assert num_instances > 0, "Need to have at least one instance"
        m, n = A[0].shape
        D_all = torch.ones((num_instances, m), device=self.device)
        E_all = torch.ones((num_instances, n), device=self.device)

        min_scale, max_scale = (1e-4, 1e4)
        n_passes = 10

        for i in range(n_passes):
            D = torch.sqrt(torch.norm(A, float('inf'), dim=2))
            E = torch.sqrt(torch.norm(A, float('inf'), dim=1))
            D[D < min_scale] = 1.0
            E[E < min_scale] = 1.0
            D[D > max_scale] = max_scale
            E[E > max_scale] = max_scale
            # assume all instances have identical cone boundaries
            start = boundaries[0, 0].clone()
            for delta in boundaries[0, 1:]:
                D[:, start:start+delta] = D[:, start:start+delta].mean(dim=1).unsqueeze(1)
                start += delta
            A = A / D.unsqueeze(2)
            A = A / E.unsqueeze(1)
            D_all *= D
            E_all *= E

        mean_row_norm = torch.norm(A, 2, dim=2).mean(dim=1)
        mean_col_norm = torch.norm(A, 2, dim=1).mean(dim=1)
        A *= scale
        return A, D_all, E_all, mean_row_norm, mean_col_norm

    def _normalize_b_c(self, b, c, D, E, mean_row_norm, mean_col_norm, scale):
        """Normalize the b, c vectors"""
        min_scale = 1e-6
        # normalize b
        b_upd = b / D
        b_upd_norm = torch.norm(b_upd, dim=1)
        b_upd_norm[b_upd_norm < min_scale] = min_scale
        sigma = mean_col_norm / b_upd_norm
        sigma = sigma.unsqueeze(1)
        b_upd = b_upd * sigma * scale
        # normalize c
        c_upd = c / E
        c_upd_norm = torch.norm(c_upd, dim=1)
        c_upd_norm[c_upd_norm < min_scale] = min_scale
        rho = mean_row_norm / c_upd_norm
        rho = rho.unsqueeze(1)
        c_upd = c_upd * rho * scale
        # return normalized b, c and sigma, rho
        return b_upd, c_upd, sigma, rho

    def scale_and_cache_all_instances_sparse(self, instances, use_scaling=True, scale=1, rho_x=1e-3):
        """Scales, computes and caches sparse LU factors of Q+I for multi-instance"""
        multi_instance = self._create_multi_instance_sparse(instances)
        if use_scaling:
            upd_multi_instance = self._scale_instances_sparse(multi_instance, scale)
        else:
            upd_multi_instance = multi_instance
        QplusI = self._compute_QplusI_sparse(upd_multi_instance, rho_x)
        upd_multi_instance.QplusI_lu = self._setup_sparse_lu_solvers(QplusI)
        upd_multi_instance.QplusI_t_lu = self._setup_sparse_lu_solvers(QplusI, transpose=True)
        return upd_multi_instance

    def _compute_QplusI(self, multi_instance, rho_x):
        """Compute the Q+I matrix"""
        A = multi_instance.A
        b = multi_instance.b.unsqueeze(2)
        c = multi_instance.c.unsqueeze(2)
        A, b, c = A.to(self.device), b.to(self.device), c.to(self.device)
        At = A.transpose(1, 2)
        bt = b.transpose(1, 2)
        ct = c.transpose(1, 2)
        zeroes_A_rows = torch.zeros(A.size(0), A.size(1),
                                    A.size(1)).double().to(self.device)
        zeroes_At_rows = torch.zeros(At.size(0), At.size(1),
                                     At.size(1)).double().to(self.device)
        zeroes = torch.zeros(A.size(0), 1, 1).double().to(self.device)
        first_block = torch.cat([zeroes_At_rows, At, c], dim=2)
        second_block = torch.cat([-A, zeroes_A_rows, b], dim=2)
        third_block = torch.cat([-ct, -bt, zeroes], dim=2)
        Q_matrix = torch.cat([first_block, second_block, third_block], dim=1)
        matrix = Q_matrix + torch.eye(Q_matrix.size(1)).to(self.device)
        num_instances, m, n = A.shape
        matrix[:, :n, :n] *= rho_x
        return matrix

    def _compute_QplusI_sparse(self, upd_multi_instance, rho_x):
        m, n, num_instances = upd_multi_instance.get_sizes()
        all_QplusI = []
        for i in range(num_instances):
            sparse_A = upd_multi_instance.A[i]
            sparse_A.eliminate_zeros()
            sparse_At = sparse_A.transpose()
            sparse_b = sp.csc_matrix(upd_multi_instance.b[i].unsqueeze(1).numpy())
            sparse_b.eliminate_zeros()
            sparse_bt = sparse_b.transpose()
            sparse_c = sp.csc_matrix(upd_multi_instance.c[i].unsqueeze(1).numpy())
            sparse_c.eliminate_zeros()
            sparse_ct = sparse_c.transpose()
            QplusI = sp.bmat([[rho_x * sp.eye(n), sparse_At, sparse_c],
                              [-sparse_A, sp.eye(m), sparse_b],
                              [-sparse_ct, -sparse_bt, sp.eye(1)]])
            all_QplusI.append(QplusI)
        return all_QplusI

    def _setup_sparse_lu_solvers(self, all_QplusI, transpose=False):
        all_sparse_solvers = []
        for i in range(len(all_QplusI)):
            curr_QplusI = all_QplusI[i]
            if transpose:
                curr_matrix = curr_QplusI.transpose()
            else:
                curr_matrix = curr_QplusI
            curr_solver = sla.splu(curr_matrix)
            all_sparse_solvers.append(curr_solver)
        return all_sparse_solvers

    def select_instances(self, multi_instance, index):
        if self.use_sparse:
            return self.select_instances_sparse(multi_instance, index)
        else:
            return self.select_instances_dense(multi_instance, index)

    def select_instances_sparse(self, multi_instance, index):
        A_select = [multi_instance.A[i] for i in index]
        b_select = multi_instance.b[index]
        c_select = multi_instance.c[index]
        cones_select = multi_instance.all_cones  # assume only one cone present, and identical
        scaled_data = None
        if hasattr(multi_instance, "D"):
            D_select = multi_instance.D[index]
            E_select = multi_instance.E[index]
            sigma_select = multi_instance.sigma[index]
            rho_select = multi_instance.rho[index]
            b_orig = multi_instance.orig_b[index]
            c_orig = multi_instance.orig_c[index]
            scaled_data = [D_select, E_select, sigma_select, rho_select,
                           b_orig, c_orig]
        upd_multi_instance = ScsMultiInstance(A_select, b_select, c_select,
                                              cones_select,
                                              scaled_data, device=self.device,
                                              use_tensors=False)
        if hasattr(multi_instance, "QplusI"):
            QplusI = [multi_instance.QplusI[i] for i in index]
            upd_multi_instance.QplusI = QplusI
        if hasattr(multi_instance, "QplusI_lu"):
            QplusI_lu_select = [multi_instance.QplusI_lu[i] for i in index]
            upd_multi_instance.QplusI_lu = QplusI_lu_select
        if hasattr(multi_instance, "QplusI_t_lu"):
            QplusI_t_lu_select = [multi_instance.QplusI_t_lu[i] for i in index]
            upd_multi_instance.QplusI_t_lu = QplusI_t_lu_select
        if hasattr(multi_instance, "soln"):
            upd_multi_instance.soln = multi_instance.soln[index]
        return upd_multi_instance

    def select_instances_dense(self, multi_instance, index):
        A_select = multi_instance.A[index]
        b_select = multi_instance.b[index]
        c_select = multi_instance.c[index]
        cones_select = [multi_instance.all_cones[i] for i in index]
        scaled_data = None
        if hasattr(multi_instance, "D"):
            D_select = multi_instance.D[index]
            E_select = multi_instance.E[index]
            sigma_select = multi_instance.sigma[index]
            rho_select = multi_instance.rho[index]
            b_orig = multi_instance.orig_b[index]
            c_orig = multi_instance.orig_c[index]
            scaled_data = [D_select, E_select, sigma_select, rho_select, 
                           b_orig, c_orig]
        upd_multi_instance = ScsMultiInstance(A_select, b_select, c_select,
                                              cones_select,
                                              scaled_data, device=self.device)
        if hasattr(multi_instance, "QplusI_t_lu"):
            QplusI_t_lu_select = [multi_instance.QplusI_t_lu[i][index] for i in range(len(multi_instance.QplusI_t_lu))]
            upd_multi_instance.QplusI_t_lu = QplusI_t_lu_select
        if hasattr(multi_instance, "soln"):
            upd_multi_instance.soln = multi_instance.soln[index]
        return upd_multi_instance

    def _create_multi_instance_sparse(self, instances):
        """Create an ScsMultiInstance with sparse A from list of instances"""
        all_cones = []
        for i in range(1):  # assumes all cones are identical
            all_cones.append(instances[i].cones)
        A = [curr_instance.data["A"] for curr_instance in instances]
        b = np.stack([curr_instance.data["b"] for curr_instance in instances])
        c = np.stack([curr_instance.data["c"] for curr_instance in instances])
        b, c = torch.from_numpy(b), torch.from_numpy(c)
        multi_instance = ScsMultiInstance(A, b, c, all_cones,
                                          device=self.device,
                                          use_tensors=False)
        return multi_instance

    def _scale_instances_sparse(self, multi_instance, scale):
        """Create the scaled instance using the original instance"""
        # assumes all cones are identical
        boundaries = ConeUtils.get_cone_boundaries(multi_instance.all_cones[0])
        A, b, c = multi_instance.A, multi_instance.b, multi_instance.c
        A_upd, D, E, mean_row_norm, mean_col_norm = self._normalize_A_sparse(
                                                    A, boundaries, scale
                                                )
        b_upd, c_upd, sigma, rho = self._normalize_b_c(
                                b, c, D, E, mean_row_norm, mean_col_norm, scale
                            )
        multi_instance = ScsMultiInstance(A_upd, b_upd, c_upd,
                                          multi_instance.all_cones,
                                          (D, E, sigma, rho, b, c),
                                          device=self.device,
                                          use_tensors=False)
        return multi_instance

    def _normalize_A_sparse(self, all_A, boundaries, scale):
        """
           Normalize the A matrix. This code comes from:
           https://github.com/bodono/scs-python/blob/master/test/test_scs_python_linsys.py
        """
        updated_A, all_instance_D, all_instance_E = [], [], []
        all_instance_row_norm, all_instance_col_norm = [], []

        for A in all_A:
            m, n = A.shape
            D_all = np.ones(m)
            E_all = np.ones(n)

            min_scale, max_scale = (1e-4, 1e4)
            n_passes = 10

            for i in range(n_passes):
                D = np.sqrt(sla.norm(A, float('inf'), axis=1))
                E = np.sqrt(sla.norm(A, float('inf'), axis=0))
                D[D < min_scale] = 1.0
                E[E < min_scale] = 1.0
                D[D > max_scale] = max_scale
                E[E > max_scale] = max_scale
                start = boundaries[0]
                for delta in boundaries[1:]:
                    D[start:start+delta] = D[start:start+delta].mean()
                    start += delta
                A = sp.diags(1/D).dot(A).dot(sp.diags(1/E))
                D_all *= D
                E_all *= E

            mean_row_norm = sla.norm(A, 2, axis=1).mean()
            mean_col_norm = sla.norm(A, 2, axis=0).mean()
            A *= scale
            updated_A.append(A)
            all_instance_D.append(D_all)
            all_instance_E.append(E_all)
            all_instance_row_norm.append(mean_row_norm)
            all_instance_col_norm.append(mean_col_norm)

        D_final = np.stack(all_instance_D)
        E_final = np.stack(all_instance_E)
        mean_row_norm = np.stack(all_instance_row_norm)
        mean_col_norm = np.stack(all_instance_col_norm)

        D_tensor, E_tensor = torch.from_numpy(D_final), torch.from_numpy(E_final)
        row_norm, col_norm = torch.from_numpy(mean_row_norm), torch.from_numpy(mean_col_norm)
        return updated_A, D_tensor, E_tensor, row_norm, col_norm

    def _compute_residuals(self, u, v, multi_instance):
        if self.use_sparse:
            return self._compute_residuals_sparse(u, v, multi_instance)
        else:
            return self._compute_residuals_dense(u, v, multi_instance)

    def _compute_residuals_dense(self, u, v, multi_instance):
        """Compute residuals"""
        A, b, c = multi_instance.A, multi_instance.b, multi_instance.c
        m, n, num_instances = multi_instance.get_sizes()
        all_tau = u[:, -1].unsqueeze(1)
        x, y = u[:, :n]/all_tau, u[:, n:n+m]/all_tau
        s = v[:, n:n+m]/all_tau

        # compute primal & dual residuals
        x_expand, y_expand = x.unsqueeze(2), y.unsqueeze(2)
        prim_res = (A @ x_expand).squeeze() + s - b
        dual_res = (A.transpose(1, 2) @ y_expand).squeeze() + c

        orig_b, orig_c = b, c

        if multi_instance.scaled:
            D, E, sigma, rho = multi_instance.D, multi_instance.E, \
                multi_instance.sigma, multi_instance.rho
            prim_res = (D * prim_res) / sigma
            dual_res = (E * dual_res) / rho
            orig_b = multi_instance.orig_b
            orig_c = multi_instance.orig_c

        prim_res = prim_res / (1 + orig_b.norm(dim=1).unsqueeze(1))
        dual_res = dual_res / (1 + orig_c.norm(dim=1).unsqueeze(1))

        return prim_res, dual_res

    def _compute_residuals_sparse(self, u, v, multi_instance):
        """Compute residuals"""
        all_A, b, c = multi_instance.A, multi_instance.b, multi_instance.c
        m, n, num_instances = multi_instance.get_sizes()
        all_tau = u[:, -1].unsqueeze(1)
        x, y = u[:, :n]/all_tau, u[:, n:n+m]/all_tau
        s = v[:, n:n+m]/all_tau

        # compute primal & dual residuals
        all_prim_res, all_dual_res = [], []
        x, y, s = x.detach().numpy(), y.detach().numpy(), s.detach().numpy()
        b_np, c_np = b.detach().numpy(), c.detach().numpy()
        for i, A in enumerate(all_A):
            prim_res = (A * x[i]) + s[i] - b_np[i]
            dual_res = (A.transpose() * y[i]) + c_np[i]
            all_prim_res.append(prim_res)
            all_dual_res.append(dual_res)
        prim_res = torch.from_numpy(np.stack(all_prim_res))
        dual_res = torch.from_numpy(np.stack(all_dual_res))

        orig_b, orig_c = b, c

        if multi_instance.scaled:
            D, E, sigma, rho = multi_instance.D, multi_instance.E, \
                multi_instance.sigma, multi_instance.rho
            prim_res = (D * prim_res) / sigma
            dual_res = (E * dual_res) / rho
            orig_b = multi_instance.orig_b
            orig_c = multi_instance.orig_c

        prim_res = prim_res / (1 + orig_b.norm(dim=1).unsqueeze(1))
        dual_res = dual_res / (1 + orig_c.norm(dim=1).unsqueeze(1))

        return prim_res, dual_res

    def _compute_residuals_sparse_for_backprop(self, u, v, multi_instance):
        """Compute residuals for backpropagation for use with regularize
            Ensure that nan never gets computed.
        """
        all_A, b, c = multi_instance.A, multi_instance.b, multi_instance.c
        m, n, num_instances = multi_instance.get_sizes()
        all_tau = u[:, -1]
        bad_tau = (all_tau <= 0)
        clean_tau = (all_tau > 0) * all_tau + 1 * bad_tau
        clean_tau = clean_tau.unsqueeze(1)
        x, y = u[:, :n]/clean_tau, u[:, n:n+m]/clean_tau
        s = v[:, n:n+m]/clean_tau

        # compute primal & dual residuals
        if hasattr(multi_instance, 'A_tensor'):
            A = multi_instance.A_tensor
        else:
            all_A_dense = np.stack([curr_A.toarray() for curr_A in all_A])
            A = torch.from_numpy(all_A_dense)
            multi_instance.A_tensor = A 

        x_expand, y_expand = x.unsqueeze(2), y.unsqueeze(2)
        prim_res = (A @ x_expand).squeeze() + s - b
        dual_res = (A.transpose(1, 2) @ y_expand).squeeze() + c

        orig_b, orig_c = b, c

        if multi_instance.scaled:
            D, E, sigma, rho = multi_instance.D, multi_instance.E, \
                multi_instance.sigma, multi_instance.rho
            prim_res = (D * prim_res) / sigma
            dual_res = (E * dual_res) / rho
            orig_b = multi_instance.orig_b
            orig_c = multi_instance.orig_c

        prim_res = prim_res / (1 + orig_b.norm(dim=1).unsqueeze(1))
        dual_res = dual_res / (1 + orig_c.norm(dim=1).unsqueeze(1))

        bad_tau_filler = bad_tau.unsqueeze(1)
        prim_res = prim_res * (~bad_tau_filler) + 0.0 * bad_tau_filler
        dual_res = dual_res * (~bad_tau_filler) + 0.0 * bad_tau_filler

        return prim_res, dual_res
    
    def _get_objectives(self, u, v, multi_instance, include_solution=False):
        """Compute objectives. Return (rescaled) solution if necessary"""
        m, n, num_instances = multi_instance.get_sizes()
        all_tau = u[:, -1].unsqueeze(1)
        x, y = u[:, :n]/all_tau, u[:, n:n+m]/all_tau
        s = v[:, n:n+m]/all_tau
        # for c, b, unlike math in SCS paper, do not take transpose
        # since the shape is already correct
        c = multi_instance.c
        b = multi_instance.b
        prim_obj = torch.bmm(c.unsqueeze(1), x.unsqueeze(2)).squeeze()
        dual_obj = -torch.bmm(b.unsqueeze(1), y.unsqueeze(2)).squeeze()

        # check if instance is scaled
        scaling_performed = multi_instance.scaled

        if scaling_performed:
            x = (x / multi_instance.E) / multi_instance.sigma
            y = (y / multi_instance.D) / multi_instance.rho
            s = (multi_instance.D * s) / multi_instance.sigma

            c = multi_instance.orig_c
            b = multi_instance.orig_b

            prim_obj = torch.bmm(c.unsqueeze(1), x.unsqueeze(2)).squeeze()
            dual_obj = -torch.bmm(b.unsqueeze(1), y.unsqueeze(2)).squeeze()

        if prim_obj.dim() == 0:
            prim_obj = prim_obj.unsqueeze(0)
        if dual_obj.dim() == 0:
            dual_obj = dual_obj.unsqueeze(0)

        if include_solution:
            return x, y, s, prim_obj, dual_obj
        return prim_obj, dual_obj

    def _convert_to_sequential_list(self, solns, metrics, num_instances):
        """Convert the solutions and result metrics to the format consistent
           with sequential SCS"""
        new_solns, new_metrics = [], []

        for i in range(num_instances):
            new_solns.append({})
        soln_keys = {}
        for key, value in solns.items():
            if value.dim() == 0:
                value = value.unsqueeze(0)
            soln_keys[key] = [value[i] for i in range(value.size(0))]
            for i in range(num_instances):
                new_solns[i][key] = soln_keys[key][i]

        if len(metrics.keys()) == 0:
            return new_solns, new_metrics

        for j in range(num_instances):
            new_metrics.append([])
            for i, key in enumerate(["residuals", "objectives"]):
                new_metrics[j].append([])
                num_iterations = len(metrics[key])
                for k in range(num_iterations):
                    new_pair = [metrics[key][k][0][j], metrics[key][k][1][j]]
                    new_metrics[j][i].append(new_pair)

        key = "all_tau"
        if key in metrics:
            num_iterations = len(metrics[key])
            for j in range(num_instances):
                new_metrics[j].append([])
                for k in range(num_iterations):
                    new_metrics[j][-1].append(metrics["all_tau"][k][j])

        key = "diffs_u"
        num_iterations = len(metrics[key])
        for j in range(num_instances):
            new_metrics[j].append([])
            for k in range(num_iterations):
                new_metrics[j][-1].append(metrics["diffs_u"][k][j])

        return new_solns, new_metrics
