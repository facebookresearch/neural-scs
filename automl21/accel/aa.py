# Copyright (c) Facebook, Inc. and its affiliates.

import torch

from .base_accel import BaseAccel

from collections import namedtuple
AAHidden = namedtuple("AAHidden", "xs gs ys")


class AA(BaseAccel):
    def __init__(self, iterate_size, context_size=None, memory_size=None):
        super().__init__(iterate_size=iterate_size, context_size=context_size)
        assert memory_size is not None
        self.memory_size = memory_size

    def init_instance(self, init_x, context=None):
        single = init_x.dim() == 1
        init_xs = [init_x.unsqueeze(0) if single else init_x]
        return init_x, AAHidden(init_xs, [], [])

    def update(self, fx, x, hidden):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
            fx = fx.unsqueeze(0)

        assert x.dim() == fx.dim() == 2
        assert x.size(0) == fx.size(0)

        g = x - fx
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
            x = x - Binv.bmm(g.unsqueeze(2)).squeeze(2)
        else:
            x = fx

        hidden.xs.append(x)

        if single:
            x = x.squeeze(0)
            g = g.squeeze(0)
        return x, g, hidden
