# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python3

import torch
from abc import abstractmethod
import numpy as np
from ..utils import MatrixUtils


class LinearOperator:
    """Base class for LinearOperators."""
    # copied from https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/operators/linear_operator.py
    # not sure what the point of such a constructor is
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @abstractmethod
    def matvec(self, rhs):
        """
        Performs a matrix-vector multiplication :math:`Kv` with the matrix
        :math:`K` that this LinearOperator represents. Should behave as
        :func:`torch.matmul`. If the LinearOperator represents a batch of
        matrices, this method should therefore operate in batch mode as well.
        """
        raise NotImplementedError("The class {} requires a matvec "
                                  "function!".format(self.__class__.__name__))


class ZeroLinearOperator(LinearOperator):
    """The LinearOperator representing zero"""
    def __init__(self, m, n):
        super().__init__(m, n)
        self.m, self.n = m, n

    def matvec(self, rhs):
        rhs_size_ind = -2 if rhs.dim() > 2 else -1
        if self.n != rhs.size(rhs_size_ind):
            raise RuntimeError("Size mismatch, self: {}, rhs: {}".format(self.n, rhs.size()))
        return rhs * 0


class IdentityLinearOperator(LinearOperator):
    """The linear operator representing the identity matrix"""
    def __init__(self, n):
        super().__init__()
        self.m, self.n = n, n

    def matvec(self, rhs):
        if self.n != rhs.size(-1):
            raise RuntimeError("Size mismatch, self: {}, rhs: {}".format(self.size(), rhs.size()))
        return rhs


class BlockDiagonalLinearOperator(LinearOperator):
    """
    The linear operator representing the block diagonal operation. Modeled on:
    https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/operators/block_diag_linear_operator.py
    """
    def __init__(self, all_linops):
        super().__init__(all_linops)
        assert len(all_linops) > 0, 'Must have at least one LinearOperator'
        self.all_linops = all_linops
        self.block_sizes = []
        total_m, total_n = 0, 0
        for curr_op in self.all_linops:
            self.block_sizes.append([curr_op.m, curr_op.n])
            total_m += curr_op.m
            total_n += curr_op.n
        self.m, self.n = total_m, total_n

    def matvec(self, rhs):
        result = torch.zeros_like(rhs)
        total_block_cols = sum([size[1] for size in self.block_sizes])
        assert total_block_cols == rhs.size(-1), (
            "The total block column length must match the non-batch dim of input"
        )
        i, j = 0, 0
        for k, curr_op in enumerate(self.all_linops):
            curr_m, curr_n = self.block_sizes[k]
            result[:, i:i+curr_m] = curr_op.matvec(rhs[:, j:j+curr_n])
            i += curr_m
            j += curr_n
        return result


class DiagonalLinearOperator(LinearOperator):
    """
    Diagonal linear operator.
    """
    def __init__(self, diag):
        super().__init__(diag)
        self.diag = diag
        self.m, self.n = diag.size(-1), diag.size(-1)

    def matvec(self, rhs):
        # TODO: make sure it batches correctly
        assert self.diag.size(-1) == rhs.size(-1)
        return self.diag * rhs


class ScalarLinearOperator(LinearOperator):
    """
    Scalar linear operator.
    """
    def __init__(self, scalar):
        super().__init__(scalar)
        self.scalar = scalar
        self.m, self.n = 1, 1

    def matvec(self, rhs):
        assert rhs.size(-1) == 1, 'Scalar operator requires rhs of size 1'
        result = self.scalar * rhs
        return result


class DProjZeroConeLinearOperator(LinearOperator):
    """
    Linear operator for dprojection into zero cone
    """
    def __init__(self, x, dual=False):
        super().__init__(dual)
        n = x.size(-1)
        self.m, self.n = n, n
        if dual:
            self.op = IdentityLinearOperator(n)
        else:
            self.op = ZeroLinearOperator(n, n)

    def matvec(self, rhs):
        return self.op.matvec(rhs)


class DProjPosConeLinearOperator(LinearOperator):
    """
    Linear operator for dprojection into positive cone
    """
    def __init__(self, x):
        super().__init__()
        n = x.size(-1)
        self.m, self.n = n, n
        diag_vals = 0.5 * (torch.sign(x) + 1)
        self.op = DiagonalLinearOperator(diag_vals)

    def matvec(self, rhs):
        return self.op.matvec(rhs)


class DProjSocConeLinearOperator(LinearOperator):
    """
    Linear operator for dprojection into SOC cone
    """
    def __init__(self, x):
        super().__init__()
        self.m, self.n = x.size(-1), x.size(-1)
        self.x = x
            
    def matvec(self, rhs):
        t = self.x[:, 0]
        z = self.x[:, 1:]
        norm_z = z.norm(dim=1)
        unit_z = z / norm_z.unsqueeze(1)

        y1_index = (norm_z <= t)
        y2_index = (norm_z <= -t)
        y_t = rhs[:, 0]
        y_z = rhs[:, 1:]
        z_dot_yz = torch.bmm(z.unsqueeze(1), y_z.unsqueeze(2)).squeeze()
        first_chunk = norm_z * y_t + z_dot_yz
        unit_z_dot_yz = torch.bmm(unit_z.unsqueeze(1), y_z.unsqueeze(2)).squeeze()
        if unit_z_dot_yz.dim() == 0:
            unit_z_dot_yz = unit_z_dot_yz.unsqueeze(0)
        a = z * y_t.unsqueeze(1)
        b = (t + norm_z).unsqueeze(1) * y_z
        c = t.unsqueeze(1) * unit_z * unit_z_dot_yz.unsqueeze(1)
        second_chunk = a + b - c
        output = torch.cat([first_chunk.unsqueeze(1), second_chunk], dim=1)
        output = (1.0 / (2 * norm_z.unsqueeze(1))) * output
        output[y1_index] = rhs[y1_index]
        output[y2_index] = 0

        return output


class DProjPsdConeLinearOperator(LinearOperator):
    """
    Linear operator for dprojection into PSD cone
    """
    def __init__(self, x):
        dim = int(np.sqrt(2 * x.size(-1)))
        X = MatrixUtils.matrix_from_lower_triangular(x, dim)
        self.m, self.n = x.size(-1), x.size(-1)  # vector length
        self.lambd_full, self.Q_full = torch.symeig(X, eigenvectors=True)
        num_neg_eig = (self.lambd_full < 0).sum(dim=1)
        self.k = num_neg_eig - 1

    def matvec(self, rhs):
        index_pos = (self.k == -1)
        index_neg = (self.k > -1)

        self.lambd, self.Q = self.lambd_full[index_neg], self.Q_full[index_neg]
        reduced_k = self.k[index_neg]
        rhs_neg = rhs[index_neg]
    
        Q_t = self.Q.transpose(1, 2)
        dim = int(np.sqrt(2 * rhs_neg.size(-1)))
        tmp = Q_t @ MatrixUtils.matrix_from_lower_triangular(rhs_neg, dim) @ self.Q
        tmp_rows, tmp_cols = tmp.size(1), tmp.size(2)

        for a in range(rhs_neg.size(0)):
            k_plus_1 = reduced_k[a] + 1
            tmp[a, 0:k_plus_1, 0:k_plus_1] = 0

            zero_tensor = torch.tensor(0, dtype=rhs.dtype, device=rhs.device)
            for i in range(k_plus_1, tmp_rows):
                for j in range(0, k_plus_1):
                    lambd_i_pos = torch.max(self.lambd[a, i], zero_tensor)
                    lambd_j_neg = -torch.min(self.lambd[a, j], zero_tensor)
                    tmp[a, i, j] *= lambd_i_pos / (lambd_i_pos + lambd_j_neg)

            for i in range(0, k_plus_1):
                for j in range(k_plus_1, tmp_cols):
                    lambd_i_neg = -torch.min(self.lambd[a, i], zero_tensor)
                    lambd_j_pos = torch.max(self.lambd[a, j], zero_tensor)
                    tmp[a, i, j] *= lambd_j_pos / (lambd_j_pos + lambd_i_neg)

        result2 = self.Q @ tmp @ Q_t
        dim = result2.size(-1)
        result_neg = MatrixUtils.lower_triangular_from_matrix(result2, dim)

        result = torch.zeros_like(rhs)
        result[index_pos] = rhs[index_pos]
        result[index_neg] = result_neg

        return result

