import numpy as np
import numba


@numba.jit(nopython=True)
def any_jitted(costs, cost):

    rows, cols = costs.shape
    ans = np.zeros(shape=rows, dtype=np.bool_)

    for i in range(rows):
        for j in range(cols):
            if costs[i, j] < cost[j]:
                ans[i] = True
                break
    return ans


@numba.jit(nopython=True)
def paretoset_jit(costs):
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
        # Keep any point with a lower cost
        nondominated_point_mask = any_jitted(costs, costs[next_point_index])
        # np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True  # And keep self

        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]

        # Re-adjust the index
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    is_efficient_mask = np.zeros(shape=n_points, dtype=np.bool_)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask
