import numpy as np
import collections


def paretoset_naive(costs):
    """Naive implementation.

    Parameters
    ----------
    costs : (np.ndarray) Array of shape (n_costs, n_objectives).

    Returns
    -------
    mask : (np.ndarray) Boolean array indicating the paretoset.

    """
    n_costs, n_objectives = costs.shape

    # Assume all points/costs are efficient
    is_efficient = np.ones(n_costs, dtype=np.bool_)

    for i in range(n_costs):
        this_cost = costs[i, :]

        # Points that are incomparable to `this_cost`, or dominated by `this_cost`.
        # In 2D, these points are above or to the right of `this_cost`.
        incomparable_or_dominated_by = np.any(costs > this_cost, axis=1)

        # Any point is incomparable to itself
        incomparable_or_dominated_by[i] = True

        # By definition: `this_cost` is efficient if all other points are incomparable or dominated by `this_cost`,
        # i.e., no other point dominates `this_cost`.
        is_efficient[i] = np.all(incomparable_or_dominated_by)
    return is_efficient


def paretoset_efficient(costs):
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
