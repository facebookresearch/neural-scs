# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python3

from abc import abstractmethod


class Solver:
    """
    Abstract Solver base class
    Args:
        how_to_batch: How to solve a batch of problems. Options are
            `simultaneous` and `sequential`
    """
    def __init__(self, how_to_batch='simultaneous'):
        self.how_to_batch = how_to_batch

    @abstractmethod
    def solve(self, prob_instances, **kwargs):
        """Abstract function that must be implemented by derived classes"""
        pass
