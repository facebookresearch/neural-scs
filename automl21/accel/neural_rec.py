# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn

from abc import abstractmethod
from .base_accel import BaseAccel
from .utils import mlp

from collections import namedtuple
NeuralLSTMHidden = namedtuple("NeuralLSTMHidden", "lstm_h lstm_c")


class NeuralRec(BaseAccel):
    def __init__(
        self, rec_class, iterate_size, context_size=None,
        rec_n_hidden=256, rec_n_layers=1,
        init_hidden_depth=0, init_hidden_n_hidden=256,
        init_act='relu', init_hidden_weight_scale=None,
        enc_hidden_depth=0, enc_n_hidden=256,
        enc_act='relu', enc_weight_scale=None,
        dec_hidden_depth=0, dec_n_hidden=256,
        dec_act='relu', dec_weight_scale=None,
        output_delta_weight=1.0,
        learn_init_iterate=False, learn_init_hidden=True,
        learn_init_iterate_delta=False,
        center_iterates=False,
    ):
        super().__init__(iterate_size=iterate_size, context_size=context_size)

        self.learn_init_iterate = learn_init_iterate
        self.learn_init_hidden = learn_init_hidden
        self.rec_n_hidden = rec_n_hidden
        self.rec_n_layers = rec_n_layers
        self.output_delta_weight = output_delta_weight
        self.learn_init_iterate_delta = learn_init_iterate_delta

        if center_iterates:
            raise NotImplementedError('center_iterates not implemented')

        is_lstm = rec_class == nn.LSTM

        if learn_init_hidden or learn_init_iterate:
            output_dim = (1 + is_lstm) * self.rec_n_layers * \
              self.rec_n_hidden * int(learn_init_hidden) + \
              iterate_size * int(learn_init_iterate)
            self.init_hidden_net = mlp(
                input_dim=context_size,
                hidden_dim=init_hidden_n_hidden,
                output_dim=output_dim,
                hidden_depth=init_hidden_depth,
                act=init_act, init_weight_scale=init_hidden_weight_scale,
            )

        self.enc = mlp(
            3*iterate_size,
            hidden_dim=enc_n_hidden,
            output_dim=self.rec_n_hidden,
            hidden_depth=enc_hidden_depth,
            act=enc_act, init_weight_scale=enc_weight_scale)

        self.cell = rec_class(
            input_size=self.rec_n_hidden, hidden_size=self.rec_n_hidden,
            num_layers=self.rec_n_layers)

        self.dec = mlp(
            input_dim=self.rec_n_hidden,
            hidden_dim=dec_n_hidden,
            output_dim=iterate_size,
            hidden_depth=dec_hidden_depth,
            act=dec_act, init_weight_scale=dec_weight_scale)


    @abstractmethod
    def init_instance(self, init_x, context=None):
        pass

    @abstractmethod
    def update(self, fx, x, hidden):
        pass

    def _extract_layered_hidden_state(self, x):
        n_batch = x.size(0)
        x_upd = x.reshape(n_batch, self.rec_n_layers, self.rec_n_hidden)
        x_upd = x_upd.transpose(1, 0).contiguous()
        return x_upd

    def _compress_layered_hidden_state(self, x):
        n_batch = x.size(1)
        x_upd = x.transpose(1, 0)
        x_upd = x_upd.reshape(n_batch, -1)
        return x_upd

    def print_grad_stats(self):
        self.print_grad_norms(self.init_hidden_net, 'init_hidden_net')
        self.print_grad_norms(self.enc, 'enc')
        self.print_grad_norms(self.cell, 'cell')
        self.print_grad_norms(self.dec, 'dec')

    def print_grad_norms(self, mod, tag):
        ps = mod.parameters()
        s = []
        for p in ps:
            s.append('{:.2e}'.format(p.grad.norm()))
        s = ', '.join(s)
        print(f'--- {tag}: [{s}]')

    def log(self, sw, step):
        if self.learn_init_hidden or self.learn_init_iterate:
            self._log(sw, self.init_hidden_net, 'init_hidden_net', step)
        self._log(sw, self.enc, 'enc', step)
        self._log(sw, self.cell, 'rec_cell', step)
        self._log(sw, self.dec, 'dec', step)

    def _log(self, sw, model, name, step):
        ps = model.parameters()
        s, grad_s = [], []
        for p in ps:
            s.append(p.data.norm())
            if hasattr(p, "grad"):
                grad_s.append(p.grad.norm())
        sw.add_histogram(name + '/weights', torch.tensor(s), step)
        if len(grad_s) > 0:
            sw.add_histogram(name + '/gradients', torch.tensor(grad_s), step)


class NeuralLSTM(NeuralRec):
    def __init__(
        self, iterate_size, context_size=None,
        rec_n_hidden=256, rec_n_layers=1,
        init_hidden_depth=0, init_hidden_n_hidden=256,
        init_act='relu', init_hidden_weight_scale=None,
        enc_hidden_depth=0, enc_n_hidden=256,
        enc_act='relu', enc_weight_scale=None,
        dec_hidden_depth=0, dec_n_hidden=256,
        dec_act='relu', dec_weight_scale=None,
        output_delta_weight=1.0,
        learn_init_iterate=False, learn_init_hidden=True,
        learn_init_iterate_delta=False,
        center_iterates=False,
        device='cpu'
    ):
        super().__init__(
            nn.LSTM, iterate_size, context_size=context_size,
            rec_n_hidden=rec_n_hidden, rec_n_layers=rec_n_layers,
            init_hidden_depth=init_hidden_depth,
            init_hidden_n_hidden=init_hidden_n_hidden,
            init_act=init_act,
            init_hidden_weight_scale=init_hidden_weight_scale,
            enc_hidden_depth=enc_hidden_depth, enc_n_hidden=enc_n_hidden,
            enc_act=enc_act, enc_weight_scale=enc_weight_scale,
            dec_hidden_depth=dec_hidden_depth, dec_n_hidden=dec_n_hidden,
            dec_act=dec_act, dec_weight_scale=dec_weight_scale,
            output_delta_weight=output_delta_weight,
            learn_init_iterate=learn_init_iterate,
            learn_init_hidden=learn_init_hidden,
            learn_init_iterate_delta=learn_init_iterate_delta,
            center_iterates=center_iterates,
        )

    def init_instance(self, init_x, context):
        single = init_x.dim() == 1
        if single:
            init_x = init_x.unsqueeze(0)
            if context is not None:
                context = context.unsqueeze(0)

        assert init_x.dim() == 2
        if context is not None:
            assert context.dim() == 2
            assert init_x.size(0) == context.size(0)

        n_batch = init_x.size(0)

        h = torch.zeros(n_batch, self.rec_n_layers * self.rec_n_hidden,
                        dtype=init_x.dtype, device=init_x.device)
        c = torch.zeros(n_batch, self.rec_n_layers * self.rec_n_hidden,
                        dtype=init_x.dtype, device=init_x.device)
        if self.learn_init_hidden or self.learn_init_iterate:
            assert context is not None
            z = self.init_hidden_net(context)
            if self.learn_init_hidden:
                if self.learn_init_iterate:
                    # need to explicitly create a sections array because
                    # rec_n_hidden can be smaller than iterate size
                    hidden_layer_product = self.rec_n_layers * self.rec_n_hidden
                    sections = [hidden_layer_product, hidden_layer_product,
                                z.size(-1) - 2 * hidden_layer_product]
                    h, c, new_init_x = z.split(sections, dim=-1)
                else:
                    h, c = z.split(self.rec_n_layers * self.rec_n_hidden, dim=-1)
            else:
                new_init_x = z

        if self.learn_init_iterate:
            if self.learn_init_iterate_delta:
                init_x = init_x + new_init_x
            else:
                init_x = new_init_x

        if single:
            init_x = init_x.squeeze(0)

        h = self._extract_layered_hidden_state(h)
        c = self._extract_layered_hidden_state(c)

        return init_x, NeuralLSTMHidden(h, c)

    def update(self, fx, x, hidden):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
            fx = fx.unsqueeze(0)

        assert x.dim() == fx.dim() == 2
        assert x.size(0) == fx.size(0)

        g = x - fx

        x_fx_g = torch.cat((x, fx, g), dim=1)
        z = self.enc(x_fx_g)
        z = z.unsqueeze(0)
        z, (h, c) = self.cell(z, (hidden.lstm_h, hidden.lstm_c))
        x = fx + self.output_delta_weight * self.dec(z).squeeze()

        if single:
            x = x.squeeze(0)
            g = g.squeeze(0)

        return x, g, NeuralLSTMHidden(h, c)


class NeuralGRU(NeuralRec):
    def __init__(
        self, iterate_size, context_size=None,
        rec_n_hidden=256, rec_n_layers=1,
        init_hidden_depth=0, init_hidden_n_hidden=256,
        init_act='relu', init_hidden_weight_scale=None,
        enc_hidden_depth=0, enc_n_hidden=256,
        enc_act='relu', enc_weight_scale=None,
        dec_hidden_depth=0, dec_n_hidden=256,
        dec_act='relu', dec_weight_scale=None,
        output_delta_weight=1.0,
        learn_init_iterate=False, learn_init_hidden=True,
        learn_init_iterate_delta=False,
        center_iterates=False,
        device='cpu',
    ):
        super().__init__(
            nn.GRU, iterate_size, context_size=context_size,
            rec_n_hidden=rec_n_hidden, rec_n_layers=rec_n_layers,
            init_hidden_depth=init_hidden_depth,
            init_hidden_n_hidden=init_hidden_n_hidden,
            init_act=init_act,
            init_hidden_weight_scale=init_hidden_weight_scale,
            enc_hidden_depth=enc_hidden_depth, enc_n_hidden=enc_n_hidden,
            enc_act=enc_act, enc_weight_scale=enc_weight_scale,
            dec_hidden_depth=dec_hidden_depth, dec_n_hidden=dec_n_hidden,
            dec_act=dec_act, dec_weight_scale=dec_weight_scale,
            output_delta_weight=output_delta_weight,
            learn_init_iterate=learn_init_iterate,
            learn_init_hidden=learn_init_hidden,
            learn_init_iterate_delta=learn_init_iterate_delta,
            center_iterates=center_iterates,
        )

    def init_instance(self, init_x, context):
        single = init_x.dim() == 1
        if single:
            init_x = init_x.unsqueeze(0)
            if context is not None:
                context = context.unsqueeze(0)

        assert init_x.dim() == 2
        if context is not None:
            assert context.dim() == 2
            assert init_x.size(0) == context.size(0)

        n_batch = init_x.size(0)

        h = torch.zeros(n_batch, self.rec_n_layers * self.rec_n_hidden,
                        dtype=init_x.dtype, device=init_x.device)
        if self.learn_init_hidden or self.learn_init_iterate:
            assert context is not None
            z = self.init_hidden_net(context)
            if self.learn_init_hidden:
                if self.learn_init_iterate:
                    # need to explicitly create a sections array because
                    # rec_n_hidden can be smaller than iterate size
                    hidden_layer_product = self.rec_n_layers * self.rec_n_hidden
                    sections = [hidden_layer_product, z.size(-1) - hidden_layer_product]
                    h, new_init_x = z.split(sections, dim=-1)
                else:
                    h = z
            else:
                new_init_x = z

        if self.learn_init_iterate:
            if self.learn_init_iterate_delta:
                init_x = init_x + new_init_x
            else:
                init_x = new_init_x

        if single:
            init_x = init_x.squeeze(0)

        h = self._extract_layered_hidden_state(h)

        return init_x, h

    def update(self, fx, x, hidden):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
            fx = fx.unsqueeze(0)

        assert x.dim() == fx.dim() == 2
        assert x.size(0) == fx.size(0)

        g = x - fx

        x_fx_g = torch.cat((x, fx, g), dim=1)
        z = self.enc(x_fx_g)
        z = z.unsqueeze(0)
        z, h = self.cell(z, hidden)
        x = fx + self.output_delta_weight * self.dec(z).squeeze()

        if single:
            x = x.squeeze(0)
            g = g.squeeze(0)

        return x, g, h
