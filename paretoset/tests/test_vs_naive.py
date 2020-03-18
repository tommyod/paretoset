#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for algorithms related to association rules.
"""

import pytest
from paretoset.algorithms_numpy import paretoset_naive, paretoset_efficient
from paretoset.algorithms_numba import paretoset_jit

import pytest
import numpy as np


seeds = list(range(100))


@pytest.mark.parametrize("seed", seeds)
def test_on_random_instances(seed):
    """
    """
    np.random.seed(seed)
    n_costs, n_objectives = np.random.randint(1, 99, size=2)
    costs = np.random.randn(n_costs, n_objectives)

    mask_naive = paretoset_naive(costs)
    mask_efficient = paretoset_efficient(costs)

    assert np.all(mask_naive == mask_efficient)

    mask_efficient_jit = paretoset_jit(costs)
    assert np.all(mask_naive == mask_efficient_jit)


if __name__ == "__main__":
    pytest.main(args=[".", "--doctest-modules", "-v"])
