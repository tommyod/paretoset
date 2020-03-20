#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for algorithms related to association rules.
"""

from paretoset.algorithms_numpy import paretoset_naive, paretoset_efficient, pareto_rank_naive, pareto_rank_NSGA2
from paretoset.algorithms_numba import paretoset_jit

import pytest
import numpy as np
import itertools


seeds = list(range(99))
dtypes = [np.float, np.int]
bools = [True, False]
paretoset_algorithms = [paretoset_naive, paretoset_efficient, paretoset_jit]


class TestParetoSetImplementations:
    @pytest.mark.parametrize("seed, dtype, distinct", itertools.product(seeds, dtypes, bools))
    def test_on_random_instances(self, seed, dtype, distinct):
        """Test that the algorithms all return the same answer."""

        # Generate a random instance
        np.random.seed(seed)
        n_costs = np.random.randint(1, 99)
        n_objectives = np.random.randint(1, 4)
        costs = np.random.randn(n_costs, n_objectives)

        # Convert to dtype, this creates some duplicates when `dtype` is integer
        costs = np.array(costs, dtype=dtype)

        # Compute the answers
        masks = [algo(costs, distinct=distinct) for algo in paretoset_algorithms]

        # At least one element must be in the mask
        assert all(np.sum(m) > 0 for m in masks)

        # Check pairwise that the answers are identical
        for m1, m2 in zip(masks[:-1], masks[1:]):
            assert np.all(m1 == m2)

    @pytest.mark.parametrize("algorithm", paretoset_algorithms)
    def test_case_distinct_1(self, algorithm):
        """Test the `distinct` parameter on a simple example"""
        costs = np.array([[1, 1], [0, 1], [0, 1]])

        ranks_naive = algorithm(costs, distinct=True)
        assert np.all(ranks_naive == np.array([False, True, False]))

        ranks_naive = algorithm(costs, distinct=False)
        assert np.all(ranks_naive == np.array([False, True, True]))

    @pytest.mark.parametrize("algorithm", paretoset_algorithms)
    def test_case_distinct_2(self, algorithm):
        """Test the `distinct` parameter on a simple example"""
        costs = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        ranks_naive = algorithm(costs, distinct=True)
        assert np.all(ranks_naive == np.array([True, False, True, False]))

        ranks_naive = algorithm(costs, distinct=False)
        assert np.all(ranks_naive == True)

    @pytest.mark.parametrize("algorithm", paretoset_algorithms)
    def test_case_distinct_3(self, algorithm):
        """Test the `distinct` parameter on a simple example"""
        costs = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0.5, 0.5], [0.5, 0.5]])

        ranks_naive = algorithm(costs, distinct=True)
        assert np.all(ranks_naive == np.array([True, False, True, False, True, False]))

        ranks_naive = algorithm(costs, distinct=False)
        assert np.all(ranks_naive == True)

    @pytest.mark.parametrize("seed, algorithm", itertools.product(seeds, paretoset_algorithms))
    def test_invariance_under_permutations(self, seed, algorithm):
        """Test that the algorithm in invariant under random permutations of data."""

        # Create some random data
        np.random.seed(seed)
        n_costs = np.random.randint(1, 99)
        n_objectives = np.random.randint(1, 4)
        costs = np.random.randint(low=-2, high=2, size=(n_costs, n_objectives))

        # Get masks
        mask_distinct = algorithm(costs, distinct=True)
        mask_unique = algorithm(costs, distinct=False)

        # Permute the data
        permutation = np.random.permutation(np.arange(n_costs))
        assert np.sum(mask_distinct) > 0
        assert np.sum(mask_unique) > 0

        # When `distinct` is set to `False`, permutation invariance should hold
        assert np.all(mask_unique[permutation] == algorithm(costs[permutation], distinct=False))

        # Equally many should me marked in the mask, regardless of `distinct` or not
        assert np.sum(mask_distinct[permutation]) == np.sum(algorithm(costs[permutation], distinct=True))
        assert np.sum(mask_unique[permutation]) == np.sum(algorithm(costs[permutation], distinct=False))

    @pytest.mark.parametrize("seed, algorithm", itertools.product(seeds, paretoset_algorithms))
    def test_equal_values(self, seed, algorithm):
        """For each group of identical data in the Pareto set: if `distinct`, 
        the first index should be True and everything else False. 
        If not `distinct`, the group should all be True.

        """

        # Generate random data
        np.random.seed(seed)
        n_costs = np.random.randint(1, 99)
        n_objectives = np.random.randint(1, 4)
        costs = np.random.randint(low=-1, high=1, size=(n_costs, n_objectives))

        indices = np.arange(n_costs)

        # Go through every cost in the Pareto set
        mask_distinct = algorithm(costs, distinct=True)
        for cost in costs[mask_distinct]:
            mask_inds = indices[np.all(cost == costs, axis=1)]

            # Only one marked
            assert np.sum(mask_distinct[mask_inds]) == 1

            # The marked element is the first one
            assert mask_distinct[mask_inds][0]

        # Go through every cost in the Pareto set
        mask_unique = algorithm(costs, distinct=False)
        for cost in costs[mask_unique]:
            mask_inds = indices[np.all(cost == costs, axis=1)]

            # All are marked
            assert np.all(mask_unique[mask_inds])

    @pytest.mark.parametrize("algorithm, distinct", itertools.product(paretoset_algorithms, bools))
    def test_no_side_effects(self, algorithm, distinct):
        """Test that the input is not mutated."""

        # Generate random data
        np.random.seed(42)
        costs = np.random.randint(low=-2, high=2, size=(99, 3))

        # Copy the data before passing into function
        costs_before = costs.copy()
        algorithm(costs, distinct=distinct)

        # The input argument is not changed/mutated in any way
        assert costs_before.shape == costs.shape
        assert np.all(costs_before == costs)


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
    pytest.main(args=[".", "--doctest-modules", "--maxfail=5", "--cache-clear", "--color", "yes", ""])

    costs = np.array([[1, 1], [0, 1], [0, 1]])

    print(costs)
    print(repr(paretoset_naive(costs, distinct=True)))

    print(repr(paretoset_naive(costs, distinct=False)))
