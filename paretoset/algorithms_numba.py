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
def any_equal_jitted(costs, cost):
    """Check if any are equal over axis 1."""

    rows, cols = costs.shape

    for i in range(rows):
        equal = True  # Assume equality
        for j in range(cols):
            if costs[i, j] != cost[j]:
                equal = False
                break  # Break out early here

        # This row in `costs` was equal to `cost`, return immediately
        if equal:
            return True

    return False


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


@numba.jit(nopython=True, fastmath=True)
def dominates(a, b, length):
    """Does a dominate b?"""
    better = False
    for i in range(length):
        a_i, b_i = a[i], b[i]

        # Worse in one dimension -> does not domiate
        # This is faster than computing `at least as good` in every dimension
        if a_i > b_i:
            return False

        # Better in at least one dimension
        if a_i < b_i:
            better = True
    return better


@numba.jit(nopython=True, fastmath=True)
def window_dominates_cost(window, cost, window_rows, window_cols):
    for i in range(window_rows):
        if dominates(window[i], cost, window_cols):
            return i
    return -1


@numba.jit(nopython=True, fastmath=True)
def cost_dominates_window(window, cost, window_rows, window_cols):
    dominated_inds = []
    for i in range(window_rows):
        if dominates(cost, window[i], window_cols):
            dominated_inds.append(i)
    return dominated_inds


@numba.jit(nopython=True)
def BNL(costs, distinct=True):
    """
    Block nested loops algorithm.
    """

    is_efficient = np.arange(costs.shape[0])
    n_costs, n_objectives = costs.shape
    num_efficient = 1  # Always put the first row in the window

    window_changed = True

    for i in range(1, n_costs):  # Skip the first row, since it's in the window

        # Get the cost for this row
        this_cost = costs[i]

        # If the window indices changed in the last iteration, get window again
        if window_changed:
            window = costs[is_efficient[:num_efficient]]
            window_rows, window_cols = window.shape
            window_changed = False

        # CASE 1 : DOES ANYTHING IN THE WINDOW DOMINATE THIS COST?
        # --------------------------------------------------------

        dom_index = window_dominates_cost(window, this_cost, window_rows, window_cols)
        # `dom_index` is the index of the first window element that dominates
        # the cost. If no window elements dominate the cost, -1 is returned.
        if dom_index >= 0:
            continue  # Window dominates cost, move on.

        # CASE 2 : DOES THIS COST DOMINATE ANYTHING IN THE WINDOW?
        # --------------------------------------------------------

        # Check if anything in the window is dominated by the point in question
        dominated_inds_window = cost_dominates_window(window, this_cost, window_rows, window_cols)
        # A point in the window is dominated, remove it
        if len(dominated_inds_window) != 0:

            # Get the original indices to remove
            to_removes = [is_efficient[k] for k in dominated_inds_window]
            for to_remove in to_removes:

                # Original indices of elements in the window
                for j, efficient in enumerate(is_efficient):

                    # Found a match, remove it
                    if efficient == to_remove:
                        # Move one to the left and decrement
                        is_efficient[j:num_efficient] = is_efficient[j + 1 : num_efficient + 1]
                        num_efficient -= 1
                        break  # Break out here

        # CASE 3 : ADD THE NEW COST TO THE WINDOW
        # ---------------------------------------

        # Add to window in all cases, except if `distinct` and it's already in the window
        if (not distinct) or (not any_equal_jitted(window, this_cost)):

            # Insert the index value of the point in the last position
            is_efficient[num_efficient] = i
            # Increment the number of efficient points
            num_efficient += 1
            window_changed = True

    bools = np.zeros(costs.shape[0], dtype=np.bool_)
    bools[is_efficient[:num_efficient]] = 1
    return bools


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

            if dominates(this_cost, other_cost, n_objectives):
                # Add `other_cost` to the set of solutions dominated by `this_cost`
                dominated_set = dominated_by[i]
                dominated_set.add(j)
            elif dominates(other_cost, this_cost, n_objectives):
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


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "--maxfail=5", "--color", "yes"])
