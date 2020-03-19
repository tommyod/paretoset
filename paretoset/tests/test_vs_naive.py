#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for algorithms related to association rules.
"""

import pytest
from paretoset.algorithms_numpy import paretoset_naive, paretoset_efficient, pareto_rank_naive, pareto_rank_NSGA2
from paretoset.algorithms_numba import paretoset_jit

import pytest
import numpy as np
import itertools


seeds = list(range(25))
dtypes = [np.float, np.int]
bools = [True, False]


class TestParetoSetImplementations:
    @pytest.mark.parametrize("seed, dtype, distinct", itertools.product(seeds, dtypes, bools))
    def test_on_random_instances(self, seed, dtype, distinct):
        """
        """
        np.random.seed(seed)
        n_costs, n_objectives = np.random.randint(1, 4, size=2)
        costs = np.random.randn(n_costs, n_objectives)

        # Convert to dtype
        costs = np.array(costs, dtype=dtype)

        mask_naive = paretoset_naive(costs, distinct=distinct)
        mask_efficient = paretoset_efficient(costs, distinct=distinct)

        assert np.all(mask_naive == mask_efficient)
        assert np.sum(mask_naive) > 0

        # mask_efficient_jit = paretoset_jit(costs)
        # assert np.all(mask_naive == mask_efficient_jit)

    @pytest.mark.parametrize("algorithm", [paretoset_naive, paretoset_efficient])
    def test_case_distinct_1(self, algorithm):
        """Test the `distinct` parameter on a simple example"""
        costs = np.array([[1, 1], [0, 1], [0, 1]])

        ranks_naive = algorithm(costs, distinct=True)
        assert np.all(ranks_naive == np.array([False, True, False]))

        ranks_naive = algorithm(costs, distinct=False)
        assert np.all(ranks_naive == np.array([False, True, True]))

    @pytest.mark.parametrize("algorithm", [paretoset_naive, paretoset_efficient])
    def test_case_distinct_2(self, algorithm):
        """Test the `distinct` parameter on a simple example"""
        costs = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        ranks_naive = algorithm(costs, distinct=True)
        assert np.all(ranks_naive == np.array([True, False, True, False]))

        ranks_naive = algorithm(costs, distinct=False)
        assert np.all(ranks_naive == True)

    @pytest.mark.parametrize("algorithm", [paretoset_naive, paretoset_efficient])
    def test_case_distinct_3(self, algorithm):
        """Test the `distinct` parameter on a simple example"""
        costs = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0.5, 0.5], [0.5, 0.5]])

        ranks_naive = algorithm(costs, distinct=True)
        assert np.all(ranks_naive == np.array([True, False, True, False, True, False]))

        ranks_naive = algorithm(costs, distinct=False)
        assert np.all(ranks_naive == True)

    @pytest.mark.parametrize("seed", seeds)
    def test_invariance_under_permutations(self, seed):
        """
        """
        np.random.seed(seed)
        n_costs = np.random.randint(1, 99)
        n_objectives = np.random.randint(1, 4)
        costs = np.random.randn(n_costs, n_objectives)

        mask_naive = paretoset_naive(costs)
        mask_efficient = paretoset_efficient(costs)

        assert np.all(mask_naive == mask_efficient)
        assert np.sum(mask_naive) > 0

        mask_efficient_jit = paretoset_jit(costs)
        assert np.all(mask_naive == mask_efficient_jit)


class TestParetoRankImplementations:
    @pytest.mark.parametrize("seed", seeds)
    def test_paretorank_on_random_instances(self, seed):
        """
        """
        np.random.seed(seed)
        n_costs, n_objectives = np.random.randint(2, 9, size=2)
        costs = np.random.randn(n_costs, n_objectives)

        ranks_naive = pareto_rank_naive(costs)
        ranks_efficient = pareto_rank_NSGA2(costs)

        assert np.all(ranks_naive == ranks_efficient)


if __name__ == "__main__":
    pytest.main(args=[".", "--doctest-modules"])

    costs = np.array([[1, 1], [0, 1], [0, 1]])

    print(costs)
    print(repr(paretoset_naive(costs, distinct=True)))

    print(repr(paretoset_naive(costs, distinct=False)))
