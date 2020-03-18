import numpy as np


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

    for i, this_cost in enumerate(costs):

        # Points that are incomparable to `this_cost`, or dominated by `this_cost`
        incomparable_or_dominated_by = np.any(costs > this_cost, axis=1)

        # Any point is incomparable to itself
        incomparable_or_dominated_by[i] = True

        # By definition: `this_cost` is efficient if all other points are incomparable or dominated by `this_cost`,
        # i.e., no other point dominates `this_cost`.
        is_efficient[i] = np.all(incomparable_or_dominated_by)
    return is_efficient


def paretoset_efficient(costs):
    """

    Parameters
    ----------
    costs

    Returns
    -------

    """
    # This code is based on the answer in:
    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    n_costs, n_objectives = costs.shape

    is_efficient = np.arange(n_costs)

    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):

        # Two points `a` and `b` are *incomparable* if neither dom(a, b) nor dom(b, a).
        # Points that are incomparable to `this_cost`, or dominate `this_cost`:
        efficient_point_mask = np.any(costs < costs[next_point_index], axis=1)

        # Any point is incomparable to itself, so keep this point
        efficient_point_mask[next_point_index] = True

        # Remove dominated points
        is_efficient = is_efficient[efficient_point_mask]
        costs = costs[efficient_point_mask]

        # Re-adjust the index
        next_point_index = np.sum(efficient_point_mask[:next_point_index]) + 1

    # Create a mask with True/False value and return it
    is_efficient_mask = np.zeros(n_costs, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


def pareto_rank_naive(costs):

    n_costs, n_objectives = costs.shape
    ranks = np.zeros(n_costs, dtype=np.int_)
    remaining = np.ones(n_costs, dtype=np.bool_)
    processed = np.zeros(n_costs, dtype=np.bool_)

    i = 1
    while np.sum(remaining) > 0:

        # Mark the costs that have rank `i`
        frontier_mask = paretoset_efficient(costs[remaining])

        # Processed costs in this iteration (not processed already, and in the frontier)
        processed[np.logical_not(processed)] = frontier_mask

        # Set the rank of the processed costs that are still remaining (remaining not updated yet)
        ranks[np.logical_and(processed, remaining)] = i

        # Update the remaining values
        remaining[remaining] = np.logical_not(frontier_mask)
        i += 1

    return ranks
