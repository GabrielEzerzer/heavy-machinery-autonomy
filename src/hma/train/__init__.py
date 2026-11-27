"""Propping environment, baseline and PPO training. Importing this package caps numerical libraries
at one thread per process unless the environment already sets a count."""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
