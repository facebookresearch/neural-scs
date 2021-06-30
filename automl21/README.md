# Copyright (c) Facebook, Inc. and its affiliates.

# Our PyTorch fixed-point acceleration library

This is in `accel` and is standalone from our experiments.

# Reproducing our ENR experiment

Running `./enr.py` will start this experiment, loading
the config from `enr.yaml` and outputting the results
and plots in the `enr_results` directory.

# Reproducing our SCS experiment.

Running `python ./scs_main.py` will start a toy Lasso
experiment, loading the config from `scs_neural/configs/lasso_toy.yaml`
and outputting the results and plots in the `exp_local`directory.

Change the config used inside `scs_main.py` from `lasso_toy.yaml` to
`lasso.yaml` to start a Lasso experiment on problems
of the same size as those in the submission. `lasso.yaml` can also
be found in `scs_neural/configs`. The results and plots are again in
`exp_local` directory.

Change the marked config values in `lasso.yaml` to their specified
values in the comments to run the same experiment as in our submission.
This experiment will take ~2 days to run to completion.

# Requirements

The following packages are required to run our code:

```
torch
numpy
scipy
matplotlib
cvxpy
tensorboard
hydra-core
pandas
```
