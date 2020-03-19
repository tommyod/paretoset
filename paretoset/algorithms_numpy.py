import numpy as np
import collections


def paretoset_naive(costs, distinct=True):
    """Naive implementation.

    Parameters
    ----------
    costs : (np.ndarray) Array of shape (n_costs, n_objectives).

    Returns
    -------
    mask : (np.ndarray) Boolean array indicating the paretoset.

    """
    n_costs, n_objectives = costs.shape

    # Assume all points/costs are inefficient
    is_efficient = np.zeros(n_costs, dtype=np.bool_)

    for i in range(n_costs):
        this_cost = costs[i, :]

        at_least_as_good = np.all(costs <= this_cost, axis=1)
        any_better = np.any(costs < this_cost, axis=1)

        dominated_by = np.logical_and(at_least_as_good, any_better)

        # If we're looking for distinct points and it's already in the
        # pareto set, disregard this value.
        if distinct and np.any(is_efficient):
            if np.any(np.all(costs[is_efficient] == this_cost, axis=1)):
                continue

        if not (np.any(dominated_by[:i]) or np.any(dominated_by[i + 1 :])):
            is_efficient[i] = True

    return is_efficient


def paretoset_efficient(costs, distinct=True):
    """An efficient vectorized algorhtm.

    This algorithm was given by Peter in this answer on Stack Overflow:
    - https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    costs = costs.copy()  # The algorithm mutates the `costs` variable, so we take a copy
    n_costs, n_objectives = costs.shape

    is_efficient = np.arange(n_costs)

    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):

        this_cost = costs[next_point_index]

        # Two points `a` and `b` are *incomparable* if neither dom(a, b) nor dom(b, a).
        # Points that are incomparable to `this_cost`, or dominate `this_cost`.
        # In 2D, these points are below or to the left of `this_cost`.
        current_efficient_points = np.any(costs < this_cost, axis=1)

        # If we're not looking for distinct, keep points equal to this cost
        if not distinct:
            no_smaller = np.logical_not(current_efficient_points)
            equal_to_this_cost = np.all(costs[no_smaller] == this_cost, axis=1)
            current_efficient_points[no_smaller] = np.logical_or(
                current_efficient_points[no_smaller], equal_to_this_cost
            )

        # Any point is incomparable to itself, so keep this point
        current_efficient_points[next_point_index] = True

        # Remove dominated points
        is_efficient = is_efficient[current_efficient_points]
        costs = costs[current_efficient_points]

        # Re-adjust the index
        next_point_index = np.sum(current_efficient_points[:next_point_index]) + 1

    # Create a boolean mask from indices and return it
    is_efficient_mask = np.zeros(n_costs, dtype=np.bool_)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


def pareto_rank_naive(costs):
    """Naive implementation of Pareto ranks."""

    n_costs, n_objectives = costs.shape
    ranks = np.zeros(n_costs, dtype=np.int_)
    remaining = np.ones(n_costs, dtype=np.bool_)
    processed = np.zeros(n_costs, dtype=np.bool_)

    current_rank = 1
    while np.sum(remaining) > 0:

        # Mark the costs that have rank `i`
        frontier_mask = paretoset_efficient(costs[remaining])

        # Processed costs in this iteration (not processed already, and in the frontier)
        processed[np.logical_not(processed)] = frontier_mask

        # Set the rank of the processed costs that are still remaining (remaining not updated yet)
        ranks[np.logical_and(processed, remaining)] = current_rank

        # Update the remaining values
        remaining[remaining] = np.logical_not(frontier_mask)
        current_rank += 1

    return ranks


import numba


@numba.jit(nopython=True, fastmath=True)
def dominates(a, b):
    """Does a dominate b?"""
    better = False
    for a_i, b_i in zip(a, b):

        # Worse in one dimension -> does not domiate
        if a_i > b_i:
            return False

        # Better in at least one dimension
        if a_i < b_i:
            better = True
    return better


# @numba.jit(nopython=True)
def pareto_rank_NSGA2(costs):
    """Algorithm from the NSGA-II paper."""
    n_costs, n_objectives = costs.shape

    dominated_by = [set() for i in range(n_costs)]

    domination_counter = np.zeros(n_costs, dtype=np.int_)
    ranks = np.zeros(n_costs, dtype=np.int_)

    domination_counter2 = np.zeros(n_costs, dtype=np.int_)
    dominated_by2 = [set() for i in range(n_costs)]

    frontier = set()
    for i in range(n_costs):
        this_cost = costs[i, :]

        dominated_by_i = np.all(this_cost >= costs, axis=1) & np.any(this_cost > costs, axis=1)
        dominated_by2[i].update(set([i for i in range(n_costs) if dominated_by_i[i]]))

        dominates_i = np.all(costs <= this_cost, axis=1) & np.any(costs < this_cost, axis=1)
        domination_counter2[i] += np.sum(dominates_i)

        for j in range(n_costs):
            other_cost = costs[j, :]

            if dominates(this_cost, other_cost):
                # Add `other_cost` to the set of solutions dominated by `this_cost`
                dominated_set = dominated_by[i]
                dominated_set.add(j)
            elif dominates(other_cost, this_cost):
                # Increment domination counter of `this_cost`
                domination_counter[i] += 1

        assert (domination_counter == domination_counter2).all()
        assert [i == j for i, j in zip(dominated_by, dominated_by2)]
        if domination_counter[i] == 0:
            ranks[i] = 1
            frontier.add(i)

    rank = 2
    while frontier:
        new_frontier = set()
        # Get all points
        for frontier_i in frontier:
            for dominated_j in dominated_by[frontier_i]:
                domination_counter[dominated_j] -= 1
                if domination_counter[dominated_j] == 0:
                    ranks[dominated_j] = rank
                    new_frontier.add(dominated_j)
        rank += 1
        frontier = new_frontier

    return ranks


def crowding_distance(costs):
    """

    Parameters
    ----------
    costs

    Returns
    -------

    """
    n_costs, n_objectives = costs.shape
    distances = np.zeros_like(costs, dtype=np.float)  # Must use floats to allow np.inf

    indices_reversed = np.zeros(n_costs, dtype=np.int)  # Used to inverse the sort

    for j in range(n_objectives):

        sorted_inds = np.argsort(costs[:, j])
        sorted_column = costs[sorted_inds, j]

        # Compute the diff of a_i as a_{i+1} - a_{i-1}
        diffs = sorted_column[2:] - sorted_column[:-2]

        # Add infinity to the bounary values
        diffs = np.concatenate((np.array([np.inf]), diffs, np.array([np.inf])))

        # Reverse the sort and add values to the indices
        indices_reversed[sorted_inds] = np.arange(n_costs)
        distances[:, j] = np.take_along_axis(diffs, indices_reversed, axis=0)

    return distances


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules"])

if __name__ == "__main__":

    costs = np.random.randn(1_00, 2)
    import time as time

    st = time.time()
    crowding_distance(costs)
    print(time.time() - st)

    print(crowding_distance(costs)[0, 0])
