#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for algorithms related to association rules.
"""

from paretoset.algorithms_numpy import paretoset_naive, paretoset_efficient, pareto_rank_naive, crowding_distance
from paretoset.algorithms_numba import paretoset_jit, BNL

import pytest
import numpy as np
import itertools


seeds = list(range(99))
dtypes = [np.float, np.int]
bools = [True, False]
paretoset_algorithms = [paretoset_naive, paretoset_efficient, paretoset_jit, BNL]
paretorank_algorithms = [pareto_rank_naive]


def generate_problem_simplex(n, d):
    """Generate D dimensional data on the D-1 dimensional simplex."""
    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    data = np.random.randn(n, d - 1)
    data = np.hstack((np.zeros(n).reshape(-1, 1), data, np.ones(n).reshape(-1, 1)))
    diffs = data[:, 1:] - data[:, :-1]
    assert diffs.shape == (n, d)
    return diffs


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

        ranks_distinct = algorithm(costs, distinct=True)
        assert np.all(ranks_distinct == np.array([False, True, False]))

        ranks_non_distinct = algorithm(costs, distinct=False)
        assert np.all(ranks_non_distinct == np.array([False, True, True]))

    @pytest.mark.parametrize("algorithm", paretoset_algorithms)
    def test_case_distinct_2(self, algorithm):
        """Test the `distinct` parameter on a simple example"""
        costs = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        ranks_distinct = algorithm(costs, distinct=True)
        assert np.all(ranks_distinct == np.array([True, False, True, False]))

        ranks_non_distinct = algorithm(costs, distinct=False)
        assert np.all(ranks_non_distinct)

    @pytest.mark.parametrize("algorithm", paretoset_algorithms)
    def test_case_distinct_3(self, algorithm):
        """Test the `distinct` parameter on a simple example"""
        costs = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0.5, 0.5], [0.5, 0.5]])

        ranks_distinct = algorithm(costs, distinct=True)
        assert np.all(ranks_distinct == np.array([True, False, True, False, True, False]))

        ranks_non_distinct = algorithm(costs, distinct=False)
        assert np.all(ranks_non_distinct)

    @pytest.mark.parametrize("algorithm", paretoset_algorithms)
    def test_case_distinct_4(self, algorithm):
        """Test the `distinct` parameter on a simple example"""
        costs = np.array([[0], [0]])

        ranks_distinct = algorithm(costs, distinct=True)
        assert np.all(ranks_distinct == np.array([True, False]))

        ranks_non_distinct = algorithm(costs, distinct=False)
        assert np.all(ranks_non_distinct)

    @pytest.mark.parametrize("seed, algorithm", itertools.product(seeds, paretoset_algorithms))
    def test_invariance_under_permutations(self, seed, algorithm):
        """Test that the algorithm in invariant under random permutations of data."""

        # Create some random data
        np.random.seed(seed)
        n_costs = np.random.randint(1, 9)
        n_objectives = np.random.randint(1, 4)
        costs = np.random.randint(low=-1, high=1, size=(n_costs, n_objectives))

        # Get masks
        mask_distinct = algorithm(costs, distinct=True)
        ranks_non_distinct = algorithm(costs, distinct=False)

        # Permute the data
        permutation = np.random.permutation(np.arange(n_costs))
        assert np.sum(mask_distinct) > 0
        assert np.sum(ranks_non_distinct) > 0

        # When `distinct` is set to `False`, permutation invariance should hold
        assert np.all(ranks_non_distinct[permutation] == algorithm(costs[permutation], distinct=False))

        # Equally many should me marked in the mask, regardless of `distinct` or not
        assert np.sum(mask_distinct[permutation]) == np.sum(algorithm(costs[permutation], distinct=True))
        assert np.sum(ranks_non_distinct[permutation]) == np.sum(algorithm(costs[permutation], distinct=False))

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
    @pytest.mark.parametrize("algorithm, seed", itertools.product(paretorank_algorithms, seeds))
    def test_paretorank_on_random_instances(self, algorithm, seed):
        """"""
        # Generate random data
        np.random.seed(42)
        costs = np.random.randint(low=-2, high=2, size=(99, 3))

        ranks = algorithm(costs)

        # Minimum rank is 1, no rank greater than the number of rows
        assert np.min(ranks) == 1
        assert np.max(ranks) <= len(ranks)

    @pytest.mark.parametrize("algorithm", paretorank_algorithms)
    def test_paretorank_on_degenerate_case(self, algorithm):
        """"""
        # Generate random data
        costs = np.ones((9, 1))

        # Ranks are 1, 1, 1, 1, ...
        ranks = algorithm(costs, distinct=False)
        assert np.all(ranks == 1)

        # Ranks are 1, 2, 3, 4, ...
        ranks = algorithm(costs, distinct=True)
        assert np.all(ranks == (np.arange(len(ranks)) + 1))

    @pytest.mark.parametrize("seed, algorithm", itertools.product(seeds, paretorank_algorithms))
    def test_invariance_under_permutations(self, seed, algorithm):
        """Test that the algorithm in invariant under random permutations of data."""

        # Create some random data
        np.random.seed(seed)
        n_costs = np.random.randint(1, 9)
        n_objectives = np.random.randint(1, 4)
        costs = np.random.randint(low=-1, high=1, size=(n_costs, n_objectives))

        # Get masks
        mask_distinct = algorithm(costs, distinct=True)
        ranks_non_distinct = algorithm(costs, distinct=False)

        # Permute the data
        permutation = np.random.permutation(np.arange(n_costs))
        assert np.min(mask_distinct) > 0
        assert np.min(ranks_non_distinct) > 0

        # When `distinct` is set to `False`, permutation invariance should hold
        assert np.all(ranks_non_distinct[permutation] == algorithm(costs[permutation], distinct=False))

        # Equally many should me marked in the mask, regardless of `distinct` or not
        assert np.sum(mask_distinct[permutation]) == np.sum(algorithm(costs[permutation], distinct=True))
        assert np.sum(ranks_non_distinct[permutation]) == np.sum(algorithm(costs[permutation], distinct=False))

    @pytest.mark.parametrize("seed, algorithm", itertools.product(seeds, paretorank_algorithms))
    def test_paretorank_on_simplex(self, seed, algorithm):
        """On a simplex every value has rank 1."""
        # Generate random data on a D-1 dimensional simplex
        np.random.seed(seed)
        n_costs = 99
        n_objectives = np.random.randint(2, 10)
        costs = generate_problem_simplex(n_costs, n_objectives)

        # Ranks are 1, 1, 1, 1, ...
        ranks = algorithm(costs, distinct=False)
        assert np.all(ranks == 1)

        # Ranks are 1, 1, 1, 1, ...
        ranks = algorithm(costs, distinct=True)
        assert np.all(ranks == 1)


class TestCrowdingDistance:
    def test_on_known_instance(self):
        """Test on a small instance computed by hand."""
        costs = np.array([[1, 2], [3, 9], [5, 3], [9, 1]], dtype=np.float)
        dist = crowding_distance(costs)
        assert np.all(dist == np.array([np.inf, np.inf, 0.8125, np.inf]))

    @pytest.mark.parametrize("seed", seeds)
    def test_invariance_under_permutations(self, seed):
        """Test that the algorithm in invariant under random permutations of data."""

        # Create some random data
        np.random.seed(seed)
        n_costs = np.random.randint(1, 9)
        n_objectives = np.random.randint(1, 4)

        # Permutation invariance does not hold if rows are duplicated
        # Therefore we use floats here
        costs = np.random.randn(n_costs, n_objectives)

        # Get masks
        distances = crowding_distance(costs)

        # Verify that one distance is returned for each row
        assert len(distances) == n_costs

        # Permute the data
        permutation = np.random.permutation(np.arange(n_costs))

        # Permutation invariance should hold
        assert np.all(distances[permutation] == crowding_distance(costs[permutation]))

    def test_no_side_effects(self):
        """Test that the input is not mutated."""

        # Generate random data
        np.random.seed(42)
        costs = np.random.randn(10, 3)

        # Copy the data before passing into function
        costs_before = costs.copy()
        crowding_distance(costs)

        # The input argument is not changed/mutated in any way
        assert costs_before.shape == costs.shape
        assert np.all(costs_before == costs)


if __name__ == "__main__":
    pytest.main(
        args=[".", "--doctest-modules", "--maxfail=5", "--cache-clear", "--color", "yes", "--durations", str(10)]
    )
