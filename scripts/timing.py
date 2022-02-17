#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:09:47 2020

@author: tommy
"""
from paretoset.algorithms_numba import BNL, paretoset_jit
from paretoset.algorithms_numpy import paretoset_efficient
import time
import numpy as np
import matplotlib.pyplot as plt

from paretoset import paretoset


def generate_problem_randn(n, d):
    """Generate Gaussian data."""
    return np.random.randn(n, d)


def generate_problem_simplex(n, d):
    """Generate D dimensional data on the D-1 dimensional simplex."""
    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    data = np.random.randn(n, d - 1)
    data = np.hstack((np.zeros(n).reshape(-1, 1), data, np.ones(n).reshape(-1, 1)))
    diffs = data[:, 1:] - data[:, :-1]
    assert diffs.shape == (n, d)
    return diffs


def get_times(observations, cost_func, algorithm, num_runs):
    """Run a function and get execution times."""

    for num_obs in observations:
        num_obs = 10**num_obs
        costs = cost_func(num_obs, objectives)
        runs = []
        for run in range(num_runs):
            start_time = time.time()
            algorithm(costs)
            elapsed = time.time() - start_time
            if elapsed > 10:
                return
            runs.append(elapsed)

        result = np.median(runs)
        yield result


observations = list(range(1, 8))
num_runs = 3

plt.figure(figsize=(9, 2.5))

objectives = 3
ax1 = plt.subplot(121)
max_times = 0
plt.title("Pareto set with {} objectives".format(objectives, num_runs))
times = list(get_times(observations, generate_problem_randn, paretoset, num_runs))
max_times = max(max_times, len(times))
plt.semilogy(observations[: len(times)], times, "-o", ms=3, label="Gaussian")

times = list(get_times(observations, generate_problem_simplex, paretoset, num_runs))
max_times = max(max_times, len(times))
plt.semilogy(observations[: len(times)], times, "-o", ms=3, label="Uniform on simplex")


plt.legend(loc="lower right").set_zorder(50)
plt.xlabel("Number of observations (rows)")
plt.ylabel("Time (seconds)")
plt.grid(True, alpha=0.5, ls="--", zorder=0)
plt.xticks(observations[:max_times], ["$10^{}$".format(i) for i in observations[:max_times]])


objectives = 9
ax2 = plt.subplot(122, sharey=ax1)
max_times = 0
plt.title("Pareto set with {} objectives".format(objectives, num_runs))
times = list(get_times(observations, generate_problem_randn, paretoset, num_runs))
max_times = max(max_times, len(times))
plt.semilogy(observations[: len(times)], times, "-o", ms=3, label="Gaussian")

times = list(get_times(observations, generate_problem_simplex, paretoset, num_runs))
max_times = max(max_times, len(times))
plt.semilogy(observations[: len(times)], times, "-o", ms=3, label="Uniform on simplex")


plt.legend(loc="lower right").set_zorder(50)
plt.xlabel("Number of observations (rows)")
plt.grid(True, alpha=0.5, ls="--", zorder=0)
plt.xticks(observations[:max_times], ["$10^{}$".format(i) for i in observations[:max_times]])


plt.tight_layout()
plt.savefig("times_pareto_set.png".format(objectives), dpi=100)
plt.show()


# =================================================================================

plt.figure(figsize=(9, 2.5))

algorithms = [paretoset_efficient, BNL, paretoset_jit]

objectives = 3
ax1 = plt.subplot(121)
max_times = 0
plt.title("Pareto set with {} objectives (Gaussian)".format(objectives, num_runs))

for algorithm in algorithms:
    times = list(get_times(observations, generate_problem_randn, algorithm, num_runs))
    max_times = max(max_times, len(times))
    plt.semilogy(observations[: len(times)], times, "-o", ms=3, label=algorithm.__name__)


plt.legend(loc="lower right").set_zorder(50)
plt.xlabel("Number of observations (rows)")
plt.ylabel("Time (seconds)")
plt.grid(True, alpha=0.5, ls="--", zorder=0)
plt.xticks(observations[:max_times], ["$10^{}$".format(i) for i in observations[:max_times]])


objectives = 3
ax2 = plt.subplot(122, sharey=ax1)
max_times = 0

plt.title("Pareto set with {} objectives (Simplex)".format(objectives, num_runs))

for algorithm in algorithms:
    times = list(get_times(observations, generate_problem_simplex, algorithm, num_runs))
    max_times = max(max_times, len(times))
    plt.semilogy(observations[: len(times)], times, "-o", ms=3, label=algorithm.__name__)


plt.legend(loc="lower right").set_zorder(50)
plt.xlabel("Number of observations (rows)")
plt.grid(True, alpha=0.5, ls="--", zorder=0)
plt.xticks(observations[:max_times], ["$10^{}$".format(i) for i in observations[:max_times]])


plt.tight_layout()
plt.savefig("times_pareto_set_implementations.png".format(objectives), dpi=100)
plt.show()
