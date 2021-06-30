# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn

from .base_accel import BaseAccel
from utils import mlp

from collections import namedtuple
NeuralAAHidden = namedtuple("NeuralAAHidden", "xs gs ys lstm_h lstm_c")


class NeuralAA(BaseAccel):
    def __init__(self, iterate_size, context_size=None, memory_size=None,
                 mlp_n_hidden=128, rec_n_hidden=256, rec_n_layers=1,
                 enc_hidden_depth=0, dec_hidden_depth=0,
                 device='cpu'):
        super().__init__(iterate_size=iterate_size, context_size=context_size)
        assert memory_size is not None
        self.memory_size = memory_size

        assert device == 'cpu'
        self.device = device
        self.rec_n_layers = rec_n_layers
        self.rec_n_hidden = rec_n_hidden

        self.enc = mlp(
            3*iterate_size,
            hidden_dim=mlp_n_hidden,
            output_dim=rec_n_hidden,
            hidden_depth=enc_hidden_depth)

        self.cell = nn.LSTM(
            input_size=rec_n_hidden, hidden_size=rec_n_hidden,
            num_layers=self.rec_n_layers)

        self.dec = mlp(
            input_dim=rec_n_hidden,
            hidden_dim=mlp_n_hidden,
            output_dim=1,
            hidden_depth=dec_hidden_depth)

    def init_instance(self, init_x, context=None):
        single = init_x.dim() == 1
        init_xs = [init_x.unsqueeze(0) if single else init_x]

        n_batch = 1 if single else init_x.size(0)
        h = torch.zeros(n_batch, self.rec_n_layers * self.rec_n_hidden,
                        dtype=init_x.dtype, device=init_x.device)
        c = torch.zeros(n_batch, self.rec_n_layers * self.rec_n_hidden,
                        dtype=init_x.dtype, device=init_x.device)
        h = self._extract_layered_hidden_state(h)
        c = self._extract_layered_hidden_state(c)
        return init_x, NeuralAAHidden(init_xs, [], [], h, c)

    def update(self, fx, x, hidden):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
            fx = fx.unsqueeze(0)

        assert x.dim() == fx.dim() == 2
        assert x.size(0) == fx.size(0)

        g = x - fx
        x_fx_g = torch.cat((x, fx, g), dim=1)
        z = self.enc(x_fx_g).unsqueeze(0)
        o, (h, c) = self.cell(z, (hidden.lstm_h, hidden.lstm_c))
        o = o.squeeze(0)
        hidden.lstm_h.data = h
        hidden.lstm_c.data = c
        alpha = self.dec(o)
        alpha = (alpha + 5.).sigmoid()

        hidden.gs.append(g)

        k = len(hidden.xs) - 1
        if k > 0:
            hidden.ys.append(g - hidden.gs[-2])

            m_k = min(k, self.memory_size)
            ST = torch.stack(hidden.xs[-m_k:], dim=1) - \
              torch.stack(hidden.xs[-m_k-1:-1], dim=1)
            S = ST.transpose(1, 2)
            Y = torch.stack(hidden.ys[-m_k:], dim=1).transpose(1, 2)
            STY = ST.bmm(Y)
            STYinv_ST = ST.solve(STY).solution
            Binv = (S - Y).bmm(STYinv_ST)
            Binv.diagonal(dim1=1, dim2=2).add_(1.)

            x = x - alpha * g - (1. - alpha)*Binv.bmm(g.unsqueeze(2)).squeeze(2)
        else:
            x = fx

        hidden.xs.append(x)

        if single:
            x = x.squeeze(0)
            g = g.squeeze(0)

        return x, g, hidden

    def _extract_layered_hidden_state(self, x):
        n_batch = x.size(0)
        x_upd = x.reshape(n_batch, self.rec_n_layers, self.rec_n_hidden)
        x_upd = x_upd.transpose(1, 0).contiguous()
        return x_upd
