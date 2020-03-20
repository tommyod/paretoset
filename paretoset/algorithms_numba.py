import numpy as np
import numba


@numba.jit(nopython=True)
def any_jitted(costs, cost):
    """Check if any are smaller over axis 1."""

    rows, cols = costs.shape
    ans = np.zeros(shape=rows, dtype=np.bool_)

    for i in range(rows):
        for j in range(cols):
            if costs[i, j] < cost[j]:
                ans[i] = True
                break  # Break out early here
    return ans


@numba.jit(nopython=True)
def all_jitted(costs, cost):
    """Check if all are equal over axis 1."""

    rows, cols = costs.shape
    ans = np.ones(shape=rows, dtype=np.bool_)

    for i in range(rows):
        for j in range(cols):
            if costs[i, j] != cost[j]:
                ans[i] = False
                break  # Break out early here

    return ans


@numba.jit(nopython=True)
def paretoset_jit(costs, distinct=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]

    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):

        this_cost = costs[next_point_index]

        # Keep any point with a lower cost
        current_efficient_points = any_jitted(costs, this_cost)
        # np.any(costs < costs[next_point_index], axis=1)
        current_efficient_points[next_point_index] = True  # And keep self

        # If we're not looking for distinct, keep points equal to this cost
        if not distinct:
            no_smaller = np.logical_not(current_efficient_points)
            equal_to_this_cost = all_jitted(costs[no_smaller], this_cost)
            current_efficient_points[no_smaller] = np.logical_or(
                current_efficient_points[no_smaller], equal_to_this_cost
            )

        # Remove dominated points
        is_efficient = is_efficient[current_efficient_points]
        costs = costs[current_efficient_points]

        # Re-adjust the index
        next_point_index = np.sum(current_efficient_points[:next_point_index]) + 1

    is_efficient_mask = np.zeros(shape=n_points, dtype=np.bool_)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask
