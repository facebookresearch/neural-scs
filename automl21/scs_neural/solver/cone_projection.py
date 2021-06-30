# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python3

import torch
import math

from .linear_operator import (BlockDiagonalLinearOperator,
                              IdentityLinearOperator,
                              ScalarLinearOperator,
                              DProjZeroConeLinearOperator,
                              DProjPosConeLinearOperator,
                              DProjSocConeLinearOperator, 
                              DProjPsdConeLinearOperator)


class ConeProjection(torch.autograd.Function):
    """Implementation builds on https://github.com/cvxgrp/diffcp/blob/master/diffcp/cones.py"""

    @staticmethod
    def initialize_cones(input_cones, input_m, input_n, use_jit=False):
        # assume all cones are identical and fixed
        ConeProjection.cones = [(key, value) for key, value in input_cones.items()]
        ConeProjection.m = input_m
        ConeProjection.n = input_n
        ConeProjection.created_jit = False
        ConeProjection.use_jit = use_jit

    @staticmethod
    def _dprojection(x, cone, dual=False):
        if cone == 'f':
            return DProjZeroConeLinearOperator(x, dual)
        elif cone == 'l':
            return DProjPosConeLinearOperator(x)
        elif cone == 'q':
            return DProjSocConeLinearOperator(x)
        elif cone == 's':
            return DProjPsdConeLinearOperator(x)
        else:
            raise NotImplementedError("Cone dprojection not implemented")

    @staticmethod
    def dprojection(v, cones, dual=False):
        offset = 0
        all_linops = []
        for cone, sz in cones:
            sz = sz if isinstance(sz, (tuple, list)) else (sz,)
            if sum(sz) == 0:
                continue
            for dim in sz:
                if cone == 's':
                    upd_dim = int(dim * (dim + 1) / 2)
                    dim = upd_dim
                elif cone in ['e', 'ep', 'ed']:
                    raise NotImplementedError("Exp cone not implemented")
                curr_linop = ConeProjection._dprojection(
                    v[:, offset:offset+dim], cone, dual
                )
                all_linops.append(curr_linop)
                offset += dim
        return BlockDiagonalLinearOperator(all_linops)

    @staticmethod
    def dpi(z, cones, m, n):
        u, v, w = z[:, :n], z[:, n:m+n], z[:, m+n:]
        eye = IdentityLinearOperator(u.size(-1))
        d_proj = ConeProjection.dprojection(v, cones, dual=True)
        last = ScalarLinearOperator((w > 0.0).double())

        all_linops = [eye, d_proj, last]

        return BlockDiagonalLinearOperator(all_linops)

    @staticmethod
    def unvec_symm(x, dim):
        """Returns a dim-by-dim symmetric matrix corresponding to `x`.
        `x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric
        matrix; the correspondence is as in SCS.
        X = [ X11 X12 ... X1k
            X21 X22 ... X2k
            ...
            Xk1 Xk2 ... Xkk ],
        where
        vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
        """
        X = torch.zeros((x.size(0), dim, dim)).to(x.device).double()
        # triu_indices gets indices of upper triangular matrix in row-major order
        idx = torch.triu_indices(dim, dim)
        col_idx, row_idx = idx[0], idx[1]  # not sure why col_idx, row_idx is inverted in diffcp code
        X[:, row_idx, col_idx] = x
        X = X + X.transpose(1, 2)
        X /= math.sqrt(2)
        diag_idx = [i for i in range(dim)]
        X[:, diag_idx, diag_idx] = torch.diagonal(X, dim1=-2, dim2=-1) * math.sqrt(2) / 2
        return X

    @staticmethod
    def vec_symm(X):
        """Returns a vectorized representation of a symmetric matrix `X`.
        Vectorization (including scaling) as per SCS.
        vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
        """
        X = X.clone()
        X *= math.sqrt(2)
        dim = X.size(-1)
        diag_idx = [i for i in range(dim)]
        X[:, diag_idx, diag_idx] = torch.diagonal(X, dim1=-2, dim2=-1) / math.sqrt(2)
        col_idx, row_idx = torch.triu_indices(dim, dim)
        Y = X[:, row_idx, col_idx]
        return Y

    @staticmethod
    def _proj(x, cone, dual=False):
        """Returns the projection of x onto a cone or its dual cone."""
        if cone == 'f':
            return x.clone() if dual else torch.zeros_like(x)
        elif cone == 'l':
            return x * (x > 0)
        elif cone == 'q':
            proj = torch.zeros_like(x)
            t = x[:, 0]
            z = x[:, 1:]
            norm_z = z.norm(dim=1)
            y1_index = (norm_z <= t + 1e-8)
            proj[y1_index] = x[y1_index]
            y2_index = (norm_z <= -t)
            proj[y2_index] = 0
            y3_index = torch.ones_like(y1_index) & (~y1_index) & (~y2_index)
            y3_interm = torch.cat([norm_z.unsqueeze(-1), z], dim=-1)
            proj[y3_index] = (0.5 * (1 + t / norm_z).unsqueeze(-1) * y3_interm)[y3_index]
            return proj
        elif cone == 's':
            dim = int(math.sqrt(2 * x.size(1)))
            X = ConeProjection.unvec_symm(x, dim)
            lambd, Q = torch.symeig(X, eigenvectors=True, upper=False)
            pos_lambd = lambd * (lambd > 0)
            Q_upd = Q * pos_lambd.unsqueeze(1)  # tested with PyTorch example
            proj = ConeProjection.vec_symm(Q_upd @ Q.transpose(1, 2))
            return proj
        else:
            raise NotImplementedError("Cone not implemented")

    @staticmethod
    def pi(x, cones, dual=False):
        projection = torch.zeros_like(x)
        offset = 0
        for cone, sz in cones:
            sz = sz if isinstance(sz, (tuple, list)) else (sz,)
            if sum(sz) == 0:
                continue
            for dim in sz:
                if cone == 's':
                    upd_dim = int(dim * (dim + 1) / 2)
                    dim = upd_dim
                elif cone in ['e', 'ep', 'ed']:
                    raise NotImplementedError("Exp cone projection not implemented")
                projection[:, offset:offset + dim] = ConeProjection._proj(
                    x[:, offset:offset + dim], cone, dual=dual)
                offset += dim
        return projection

    @staticmethod
    def create_jit(u, v, w):
        def _run_cone_projection(u, v, w):
            proj_v = ConeProjection.pi(v, ConeProjection.cones, dual=True)
            w_upd = w * (w > 0.0)
            proj_z = torch.cat([u, proj_v, w_upd], dim=1)
            return proj_z
        ConeProjection.run_jitted_projection = torch.jit.trace(
            _run_cone_projection, (u, v, w))
        ConeProjection.created_jit = True

    @staticmethod
    def forward(ctx, z, cones, m, n):
        ctx.cones = cones
        ctx.save_for_backward(z)

        batched = True
        if z.dim() == 1:
            z = z.unsqueeze(0)
            cones = [cones]
            batched = False

        num_instances = z.size(0)
        ctx.sizes = (m, n, num_instances)

        # assume all cones are identical
        u, v, w = z[:, :n], z[:, n:m+n], z[:, m+n:]
        if ConeProjection.use_jit:
            if ConeProjection.created_jit is False:
                ConeProjection.create_jit(u, v, w)
            proj_z = ConeProjection.run_jitted_projection(u, v, w)
        else:
            cone_tuple = [(key, value) for key, value in cones[0].items()]
            proj_v = ConeProjection.pi(v, cone_tuple, dual=True)
            w_upd = w * (w > 0.0)
            proj_z = torch.cat([u, proj_v, w_upd], dim=1)

        if not batched:
            proj_z = proj_z.squeeze()
        return proj_z

    @staticmethod
    def backward(ctx, dproj_z):
        z, = ctx.saved_tensors
        cones = ctx.cones
        m, n, num_instances = ctx.sizes

        batched = z.dim() > 1
        if z.dim() == 1:
            z = z.unsqueeze(0)
            dproj_z = dproj_z.unsqueeze(0)
            cones = [cones]

        # assume all cones are identical
        cone_tuple = [(key, value) for key, value in cones[0].items()]
        dpi_result = ConeProjection.dpi(z, cone_tuple, m, n)
        grad = dpi_result.matvec(dproj_z)

        if not batched:
            grad = grad.squeeze()

        return grad, None, None, None
