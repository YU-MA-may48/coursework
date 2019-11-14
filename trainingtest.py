#!/usr/bin/env python
# coding:utf-8

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

import numpy as np

X = 2 * np.random.rand(100, 1)    # uniformly generate 100 random numbers, i.e., 100 rows and 1 column
y = 4 + 3 * X + np.random.randn(100, 1)   # a linear function y = 4 + 3 * X, plus Gaussian noise N(0, 1) (normal distribution)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])

plt.show()

