# Licensing 

The majority of neural-scs is licensed under the [CC
BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/),
however, portions of the project are available under separate license
terms: SCS is licensed under MIT license.

# Neural Fixed-Point Acceleration for SCS

We present neural fixed-point acceleration, a framework to
automatically learn to accelerate convex fixed-point problems that are
drawn from a distribution, using ideas from meta-learning and
classical acceleration algorithms.  We apply our framework to SCS, the
state-of-the-art solver for convex cone programming. Our work brings
neural acceleration into any optimization problem expressible with
CVXPY.


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
