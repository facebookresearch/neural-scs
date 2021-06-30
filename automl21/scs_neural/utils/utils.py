# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python3

import torch
from torch import nn

import math


class ConeUtils:

    @staticmethod
    def get_cone_boundaries(cones, combined=True):
        """Returns boundaries of different cones in input cone dictionary"""
        boundaries = []
        for cone_id in ['f', 'l', 'q', 's', 'ep']:
            if cone_id in ['f', 'l', 'ep']:
                if cones[cone_id] != 0:
                    factor = 3 if cone_id == 'ep' else 1
                    new_bound = factor * cones[cone_id]
                    if combined:
                        boundaries.append(new_bound)
                    else:
                        boundaries.append([new_bound])
            if cone_id in ['q']:
                constraints = cones[cone_id]
                if combined:
                    boundaries += constraints
                else:
                    boundaries.append(constraints)
            if cone_id in ['s']:
                if len(cones[cone_id]) > 0:
                    k = cones[cone_id][0]
                    new_bound = int(k * (k+1) / 2)
                    if combined:
                        boundaries.append(new_bound)
                    else:
                        boundaries.append([new_bound])
        return boundaries

    @staticmethod
    def block_cones(cones, nbatch):
        """Converts a list of cones into a single cone cross product"""
        block_cones = {}
        for k in ['f', 'l', 'q', 's', 'ep']:
            if k in ['f', 'l', 'ep']:
                block_cones[k] = nbatch * cones[k]
            elif k in ['q', 's']:
                block_cones[k] = []
                for sz in cones[k]:
                    block_cones[k] += [sz] * nbatch
            else:
                raise NotImplementedError
        return block_cones

    @staticmethod
    def coalesce_batch_cone_data(b, all_cones):
        """Coalesce batched vector b and cone list all_cones into a 
           single vector and cone"""
        nbatch, ncon = b.shape
        perm_b = []
        start_row = 0

        # assume cones are identical
        boundaries = ConeUtils.get_cone_boundaries(all_cones[0], combined=False) 

        for szs in boundaries:
            for sz in szs:
                assert sz > 0
                end_row = start_row + sz
                for i in range(nbatch):
                    perm_b.append(b[i][start_row:end_row])
                start_row = end_row

        assert start_row == ncon
        perm_b = torch.cat(perm_b)

        block_cones = ConeUtils.block_cones(all_cones[0], nbatch)
        return perm_b, block_cones

    @staticmethod
    def uncoalesce_projection(y, cones, nbatch):
        """Uncoalesce vector y into a batched vector"""
        y_batch = [[] for _ in range(nbatch)]
        start = 0

        # assume cones are identical
        boundaries = ConeUtils.get_cone_boundaries(cones[0], combined=False) 
        for szs in boundaries:
            for sz in szs:
                for i in range(nbatch):
                    y_batch[i].append(y[start:start+sz])
                    start += sz

        y_batch = [torch.cat(yi) for yi in y_batch]
        y_stack = torch.stack(y_batch)
        return y_stack


class MatrixUtils:

    @staticmethod
    @torch.jit.script
    def matrix_from_lower_triangular(x: torch.Tensor, dim: int) -> torch.Tensor:
        X = torch.zeros(x.size(0), dim, dim).double().to(x.device)
        idx = torch.tril_indices(dim, dim)
        row_idx, col_idx = idx[0], idx[1]
        # place the same values in both X[row, col] and X[col, row]
        X[:, row_idx, col_idx] = x
        X[:, col_idx, row_idx] = x
        # divide everything by sqrt(2)
        X /= math.sqrt(2)
        # multiply diagonals back by sqrt(2)
        diag_values = torch.diagonal(X, dim1=-2, dim2=-1) * math.sqrt(2)
        diag_mask = torch.eye(diag_values.size(-1), device=diag_values.device)
        X = diag_values.unsqueeze(2) * diag_mask + X * (1 - diag_mask)
        return X

    @staticmethod
    @torch.jit.script
    def lower_triangular_from_matrix(X: torch.Tensor, dim: int) -> torch.Tensor:
        X = X * math.sqrt(2)
        diag_values = torch.diagonal(X, dim1=-2, dim2=-1) / math.sqrt(2)
        diag_mask = torch.eye(diag_values.size(-1), device=diag_values.device)
        X = diag_values.unsqueeze(2) * diag_mask + X * (1 - diag_mask)
        idx = torch.tril_indices(dim, dim)
        row_idx, col_idx = idx[0], idx[1]
        Y = X[:, row_idx, col_idx]
        return Y


def mlp(input_dim,
        hidden_dim,
        output_dim,
        hidden_depth,
        output_mod=None,
        act=nn.ReLU,
        init_weight_scale=None):
    if isinstance(act, str):
        if act == 'relu':
            act = nn.ReLU
        elif act == 'elu':
            act = nn.ELU
        elif act == 'tanh':
            act = nn.Tanh
        else:
            raise NotImplementedError()
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), act()]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), act()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    init_weight_scale = None if init_weight_scale == "None" else init_weight_scale
    if init_weight_scale is not None:
        for mod in trunk.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.data.div_(init_weight_scale)
                mod.bias.data.zero_()
    return trunk
