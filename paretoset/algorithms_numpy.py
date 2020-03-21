import numpy as np
import paretoset.user_interface  # Import like this to avoid circular import issues


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

        # Here `NOT(ANY(a_i > b_i))` is equal to ALL(a_i <= b_i), but faster.
        at_least_as_good = np.logical_not(np.any(costs > this_cost, axis=1))
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


def pareto_rank_naive(costs, distinct=True, use_numba=True):
    """Naive implementation of Pareto ranks."""

    n_costs, n_objectives = costs.shape
    ranks = np.zeros(n_costs, dtype=np.int_)
    remaining = np.ones(n_costs, dtype=np.bool_)
    processed = np.zeros(n_costs, dtype=np.bool_)

    current_rank = 1
    while np.sum(remaining) > 0:

        # Mark the costs that have rank `i`
        frontier_mask = paretoset.user_interface.paretoset(costs[remaining], distinct=distinct, use_numba=use_numba)

        # Processed costs in this iteration (not processed already, and in the frontier)
        processed[np.logical_not(processed)] = frontier_mask

        # Set the rank of the processed costs that are still remaining (remaining not updated yet)
        ranks[np.logical_and(processed, remaining)] = current_rank

        # Update the remaining values
        remaining[remaining] = np.logical_not(frontier_mask)
        current_rank += 1

    return ranks


def crowding_distance(costs):
    """Compute the crowding distance of (non-NaN) numerical data.

    The input data in `costs` must be a NumPy ndarray of shape
    (observations, objectives). The user is responsible for dealing with NaN
    values, duplicate rows and constant columns *before* calling this function.

    Parameters
    ----------
    costs : np.ndarray
        Numerical data of shape (observations, objectives).

    Returns
    -------
    distances : np.ndarray
        The crowding distance of each observation (row).

    Examples
    --------
    >>> import numpy as np
    >>> costs = np.array([1, 3, 5, 9, 11]).reshape(-1, 1)
    >>> crowding_distance(costs)
    array([inf, 0.4, 0.6, 0.6, inf])
    """
    msg = "The `costs` argument must be np.ndarray of shape (observations, objectives)"
    if not isinstance(costs, np.ndarray):
        raise ValueError(msg)
    if not costs.ndim == 2:
        raise ValueError(msg)

    # =============================================================================
    # SETUP ARRAYS USED FOR COMPUTATION
    # =============================================================================

    # Get the shape of the inputs
    n_costs, n_objectives = costs.shape
    if n_costs <= 2:
        return np.ones(n_costs) * np.inf

    # Prepare the distance matrix
    distances = np.zeros_like(costs, dtype=np.float)  # Must use floats to allow np.inf

    # Used several times below, so pre-compute it here
    arange_objectives = np.arange(n_objectives)

    # Used to inverse the sort
    sorted_inds_reversed = np.zeros(costs.shape, dtype=np.int)

    # =============================================================================
    # PERFORM THE COMPUTATION
    # =============================================================================

    # Sort every column individually
    sorted_inds = np.argsort(costs, axis=0)
    sorted_matrix = np.take_along_axis(costs, sorted_inds, axis=0)

    # Compute the diff of a_i as a_{i+1} - a_{i-1}
    diffs = sorted_matrix[2:, :] - sorted_matrix[:-2, :]

    # Compute max minus min for each column, using the already sorted matrix
    max_minus_min = sorted_matrix[-1, :] - sorted_matrix[0, :]

    # Bottom and top rows are set to infinity - these points have 1 neighbor
    infinity_row = (np.ones(n_objectives) * np.inf).reshape(1, -1)

    # Create the full matrix of differences
    diffs = np.vstack((infinity_row, diffs, infinity_row))
    assert diffs.shape == costs.shape

    # Prepare a matrix of reverse-sorted indices
    index = sorted_inds, arange_objectives

    # Equivalent to: np.outer(np.arange(n_costs), np.ones(n_objectives))
    sorted_inds_reversed[index] = np.tile(np.arange(n_costs), (n_objectives, 1)).T

    # Distances (diffs) in original data order
    distances = diffs[sorted_inds_reversed, arange_objectives]

    # Divide each column by MAX minus MIN to normalize
    distances = np.divide(distances, max_minus_min, out=distances)

    # Sum over each row, divide by number of columns and return
    return distances.sum(axis=1) / n_objectives


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules", "--color", "yes"])

if __name__ == "__main__":

    pass
