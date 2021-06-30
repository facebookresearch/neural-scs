# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn

from abc import ABC, abstractmethod


class BaseAccel(nn.Module, ABC):
    def __init__(self, iterate_size, context_size=None, **kwargs):
        super().__init__()
        self.iterate_size = iterate_size
        self.context_size = context_size

    @abstractmethod
    def init_instance(self, init_x, context=None):
        pass

    @abstractmethod
    def update(self, fx, x, hidden):
        pass
