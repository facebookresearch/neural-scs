# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn

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
