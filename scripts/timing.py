#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:09:47 2020

@author: tommy
"""

import numpy as np


import numpy as np
import numba

from paretoset.algorithms_numpy import skyline_naive, skyline_efficient
from paretoset.algorithms_numba import skyline_efficient_jit


def is_pareto_efficient2(costs):
    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    is_efficient = np.ones(costs.shape[0], dtype=np.bool_)
    n_points = costs.shape[0]

    # print(costs)

    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < n_points:
        # print(next_point_index)

        is_efficient[is_efficient] = np.any(costs[is_efficient] < costs[next_point_index], axis=1)
        is_efficient[next_point_index] = True  # And keep self

        # print(is_efficient[next_point_index+1:])

        if not np.any(is_efficient[next_point_index + 1 :]):
            break
        next_point_index = np.argmax(is_efficient[next_point_index + 1 :]) + next_point_index + 1

    # print(is_efficient)
    return is_efficient


@numba.jit(nopython=True)
def first_true(array):
    for i in range(len(array)):
        if array[i]:
            return i
    return -1


# @numba.jit(nopython=True)
# def BNL()


def BNL_numpy(costs):
    """
    Block nested loops.
    """
    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    is_efficient = np.zeros(costs.shape[0], dtype=np.bool)
    n_points = costs.shape[0]
    indices = np.arange(n_points)

    for i, cost in enumerate(costs):

        window = costs[is_efficient]

        # Check if anything in the window dominates the point
        dominated_by_window = np.any(np.all(window < cost, axis=1))
        if dominated_by_window:
            # print(cost, "dominated by something in ")
            # print(window)
            continue

        # Check if anything in the window is dominated by the point
        window_dominated_by_point = np.all(cost < window, axis=1)
        if np.any(window_dominated_by_point):
            # print(cost, "dominates")
            # print(window[window_dominated_by_point])
            # print(is_efficient)

            inds_to_remove = indices[is_efficient][window_dominated_by_point]

            is_efficient[inds_to_remove] = 0

            # is_efficient[is_efficient[window_dominated_by_point]] = False # Remove these

        # Insert the point
        is_efficient[i] = 1

    return is_efficient


@numba.jit(nopython=True, fastmath=True)
def dominates(a, b, size):
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
def BNL_numba(costs, is_efficient):
    """
    Block nested loops.
    """
    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    # is_efficient = np.zeros(costs.shape[0], dtype=np.bool)
    n_points = costs.shape[0]
    num_efficient = 0

    window_changed = True

    for i in range(n_points):

        # print()
        # print(f"Main iteration {i}: {costs[i]}")

        cost = costs[i]
        if window_changed:
            window = costs[is_efficient[:num_efficient]]
            window_rows, window_cols = window.shape
            window_changed = False

        # Check if anything in the window dominates the point
        dom_index = window_dominates_cost(window, cost, window_rows, window_cols)
        if dom_index >= 0:
            continue
        if dom_index > 0:
            # Swap to first position
            # is_efficient[0], is_efficient[dom_index] = is_efficient[dom_index], is_efficient[0]
            # window_changed = True
            # Swap one up
            # is_efficient[dom_index-1], is_efficient[dom_index] = is_efficient[dom_index], is_efficient[dom_index-1]

            continue

        # Check if anything in the window is dominated by the point
        dominated_inds = cost_dominates_window(window, cost, window_rows, window_cols)

        if len(dominated_inds) > 0:

            # print("To remove:", dominated_inds)

            # print()
            # print("Removing from:", is_efficient[:num_efficient])

            # remove_from = is_efficient[:num_efficient].copy()
            to_removes = [is_efficient[k] for k in dominated_inds]
            for to_remove in to_removes:
                # to_remove = is_efficient[k]
                for j, efficient in enumerate(is_efficient):
                    if efficient == to_remove:
                        # print(f"Removing {to_remove} from {is_efficient[:num_efficient]}")
                        # Move one to the left and decrement
                        is_efficient[j:num_efficient] = is_efficient[j + 1 : num_efficient + 1]
                        num_efficient -= 1
                        break

            # print("Removed from:", is_efficient[:num_efficient])
            # print()

        # Insert the point
        # print()
        # print(f'Adding {i} to {is_efficient[:num_efficient]}')
        is_efficient[num_efficient] = i
        num_efficient += 1
        # print(f'Added {i} to {is_efficient[:num_efficient]}')
        window_changed = True

    return is_efficient[:num_efficient]


@numba.jit(nopython=True)
def BNL(costs):
    """
    Block nested loops.
    """
    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    is_efficient = np.arange(costs.shape[0])
    ans = BNL_numba(costs, is_efficient)
    bools = np.zeros(costs.shape[0])
    bools[ans] = 1
    return bools


def sort(costs):
    np.sort(costs)
    return None


def generate_problem_randn(n, d):
    return np.random.randn(n, d)


def generate_problem_increasing(n, d):
    return np.vstack([np.arange(n) / n] * d).T


def generate_problem_decreasing(n, d):
    return np.vstack([np.arange(n)[::-1] / n] * d).T


def generate_problem_simplex(n, d):
    x = np.linspace(0, 1, num=int(n ** (1 / d)))
    return cartesian([x] * d)


def cartesian(arrays):
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import time

    np.random.seed(123)

    algorithms = [skyline_naive, skyline_efficient, is_pareto_efficient2, BNL, skyline_efficient_jit]

    costs = np.array([[2, 1], [1, 2], [1.5, 1.5]])

    for seed in range(50):

        np.random.seed(seed)
        costs = generate_problem_randn(5, d=2).round(2)
        print(costs)

        for algorithm in algorithms:
            ans = algorithm(costs)
            print(algorithm.__name__, ans)

        solutions = [a(costs) for a in algorithms]
        assert all([np.all(solutions[0] == s) for s in solutions])

    if True:

        for algorithm in algorithms:

            n = []
            times = []
            for n_i in list(range(10)):
                costs = generate_problem_randn(10 ** (n_i), d=3)
                start_time = time.time()
                ans = algorithm(costs)
                run_time = time.time() - start_time
                times.append(run_time)
                n.append(n_i)
                if run_time > 0.5 and n_i > 2:
                    break

            plt.semilogy(n, times, label=algorithm.__name__)

        plt.grid()
        plt.legend()
        plt.show()

    if False:
        from paretoset.algorithms_numpy import pareto_rank_naive

        for algorithm in [pareto_rank_naive]:

            n = []
            times = []
            for n_i in list(range(8)):
                costs = generate_problem_randn(10 ** (n_i), d=3)
                inds = np.argsort(-costs.sum(axis=1))
                # costs = costs[inds]
                start_time = time.time()
                ans = algorithm(costs)
                run_time = time.time() - start_time
                times.append(run_time)
                n.append(n_i)
                if run_time > 5 and n_i > 2:
                    break

            plt.semilogy(n, times, label=algorithm.__name__)

        plt.grid()
        plt.legend()
        plt.show()

    if False:

        for n_i in list(range(3)):
            costs = generate_problem_randn(10 ** n_i, d=10)
            print(costs)
            solutions = [a(costs) for a in algorithms]
            print(solutions[0])
            print(solutions[-1])
            assert all([np.all(solutions[0] == s) for s in solutions])

            solution_points = costs[solutions[0], :]
            plt.scatter(costs[:, 0], costs[:, 1])
            plt.scatter(solution_points[:, 0], solution_points[:, 1], s=10)

            plt.show()

    if False:
        costs = generate_problem_randn(10 ** 2, d=2)

        from paretoset.algorithms_numpy import pareto_rank_naive

        ranks = pareto_rank_naive(costs)
        print(ranks)

        for i in range(1, np.max(ranks) + 1):
            mask = ranks == i
            plt.scatter(costs[mask, 0], costs[mask, 1], label=str(i), alpha=1)

        plt.scatter(costs[:, 0], costs[:, 1], alpha=0.5, s=80, marker="x")
        plt.grid()
        plt.legend()
        plt.show()
