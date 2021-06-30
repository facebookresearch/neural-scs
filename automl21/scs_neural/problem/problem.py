# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python3

from abc import abstractmethod
import cvxpy as cp
import itertools
import numpy as np
import torch


class ScsInstance:
    """ Class that transforms a problem in SCS format output by cvxpy
        (i.e. a ParamConeProg) into an object whose components are
        directly usable by SCS
    """
    id_iter = itertools.count()

    def __init__(self, input_problem, cones=None):
        data = {
            'A': input_problem['A'],
            'b': input_problem['b'],
            'c': input_problem['c'],
        }
        self.data = data

        if 'dims' not in input_problem:
            assert cones is not None, ("Either the problem must contain cones"
                                       "or the cones must be provided")
        else:
            cone_dims = input_problem['dims']
            cones = {
                "f": cone_dims.zero,
                "l": cone_dims.nonneg,  # needed to swap from nonpos to nonneg
                "q": cone_dims.soc,
                "ep": cone_dims.exp,
                "s": cone_dims.psd,
            }
        self.cones = cones
        self.instance_id = next(self.id_iter)

    def get_sizes(self):
        """Get the A matrix sizes for SCS problem"""
        sizes = self.data['A'].shape
        return sizes

    @staticmethod
    def create_scaled_instance(A, b, c, D, E, sigma, rho, cones):
        """
           Create new SCS instance with the provided A, b, c, and cones.
           Assumes that A, b, and c are already normalized as needed.
           Track D, E, sigma, rho to allow for rescaling the solution.
        """
        prob = {}
        prob["A"], prob["b"], prob["c"] = A, b, c
        instance = ScsInstance(prob, cones)
        instance.D, instance.E = D, E
        instance.sigma, instance.rho = sigma, rho
        return instance


class ScsMultiInstance:
    """ Class that transforms a list of problems into
         a batched set of ScsInstance data.
    """
    def __init__(self, A, b, c, cones, scaled_data=None, use_tensors=True, 
                 verify_sizes=False, device='cpu'):
        self.A, self.b, self.c = A, b, c
        self.all_cones = cones
        self.num_instances = len(A)
        if verify_sizes:
            raise NotImplementedError("Complete verification not implemented yet")
        if scaled_data is not None:
            D, E, sigma, rho, orig_b, orig_c = scaled_data
            self.D, self.E = D, E
            self.sigma, self.rho = sigma, rho
            self.orig_b, self.orig_c = orig_b, orig_c
            self.scaled = True
        else:
            
            self.scaled = False
        if use_tensors:
            self.convert_to_tensors(device)

    def get_sizes(self):
        """Get the A matrix sizes and number of instances for SCS problem"""
        assert self.num_instances > 0
        m, n = self.A[0].shape
        return m, n, self.num_instances

    def convert_to_tensors(self, device='cpu'):
        attr_list = ['A', 'b', 'c', 'D', 'E', 'sigma', 'rho', 'orig_b', 'orig_c']
        for attr in attr_list:
            if hasattr(self, attr):
                if torch.is_tensor(getattr(self, attr)):
                    continue
                value = torch.from_numpy(getattr(self, attr))
                value_device = value.to(device)
                setattr(self, attr, value_device)

    def add_solutions(self, solns):
        self.soln = solns

class Problem:
    """Abstract Problem class for extracting problems in SCS format"""
    def __init__(self, config_file=None):
        self.config_file = config_file

    @abstractmethod
    def _sample_from_distributions(self, **kwargs):
        """Samples data from specified distribution to create problem instances"""
        pass

    def write_to_file(self, output_file):
        """Writes SCS format data to file."""
        pass

    def get_sizes(self, verify_sizes=False):
        """Gets the SCS instance problem sizes"""
        if hasattr(self, 'instances') is False or len(self.instances) < 1:
            raise RuntimeError("No SCS instances to get size")
        if verify_sizes: 
            raise NotImplementedError("Complete verification not implemented yet")
        return self.instances[0].get_sizes()

    def get_cone(self):
        return self.instances[0].cones


class Lasso(Problem):
    """
    Constructs the Lasso problem from sampled distributions or data
    CVXPY problem format is taken from https://www.cvxpy.org/examples/machine_learning/lasso_regression.html
    Args:
        config_file: Configuration file
        n: Size of the regression variable
        n_samples: Number of problem instances that need to be generated
        train_data_size: Number of training samples required for the Lasso problem
    """
    def __init__(self, n, config_file=None, n_samples=10, train_data_size=100, create_data=True):
        super().__init__(config_file)
        if create_data:
            create_scs_format = self._construct_generic(n, train_data_size)
            self.instances = self._sample_from_distributions(create_scs_format, train_data_size, n, n_samples)

    @staticmethod
    def create_sampled_object(instances):
        """Create a wrapper object around the provided instances"""
        sampled_lasso = Lasso(-1, create_data=False)
        sampled_lasso.instances = instances
        return sampled_lasso

    def _construct_generic(self, n, train_data_size):
        """ Constructs a generic creator of SCS format problems for Lasso"""
        _beta = cp.Variable(n)
        _lambd = cp.Parameter(nonneg=True)
        _X = cp.Parameter((train_data_size, n))
        _Y = cp.Parameter(train_data_size)
        objective_fn = cp.norm2(_X @ _beta - _Y)**2 + _lambd * cp.norm1(_beta)
        prob = cp.Problem(cp.Minimize(objective_fn))

        def create_scs_format(X_train, Y_train, lambd):
            _X.value = X_train
            _Y.value = Y_train
            _lambd.value = lambd
            scs_prob, _, _, = prob.get_problem_data(cp.SCS)
            return scs_prob

        return create_scs_format

    def _sample_from_distributions(self, create_scs_format, m, n, n_samples=10, scs_paper=True):
        """
           Samples data from specified distribution to construct Lasso problem
           instances.
        """
        instances = []
        # Only available format right now is Lasso instances
        if scs_paper:
            for i in range(n_samples):
                X, Y, lambd = self._generate_data_scs_solver_paper(m=m, n=n)
                curr_prob = create_scs_format(X, Y, lambd)
                scs_prob = ScsInstance(curr_prob)
                instances.append(scs_prob)
        else:
            raise NotImplementedError("Only scs format data implemented")      
        return instances

    def _generate_data_scs_solver_paper(self, m=4, n=20, sigma=0.1,
                                        density=0.1):
        """Generates data for Lasso regression solver as used by the SCS paper."""
        beta_star = np.random.randn(n)
        idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
        for idx in idxs:
            beta_star[idx] = 0
        X = np.random.randn(m, n)
        Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
        lambd = 0.1 * np.linalg.norm((X.transpose() @ Y), np.inf)
        return X, Y, lambd

    def mse(X, Y, beta):
        """Computes the mean-square error for input X, Y, beta"""
        loss = np.linalg.norm((X @ beta - Y), 2)**2
        return (1.0 / X.shape[0]) * loss
