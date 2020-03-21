import importlib
import numpy as np
import collections.abc


def user_has_package(package_name):
    """Check if the user has `numba` installed."""
    try:
        importlib.import_module(package_name, package=None)
        return True
    except ModuleNotFoundError:
        return False


def validate_inputs(costs, sense=None):
    """Sanitize user inputs for the `paretoset` function.

    Examples
    --------
    >>> costs, sense = validate_inputs([1, 2, 3])
    >>> costs
    array([[1],
           [2],
           [3]])

    """

    # The input is an np.ndarray
    if isinstance(costs, np.ndarray):
        if costs.ndim == 1:
            return validate_inputs(costs.copy().reshape(-1, 1), sense=sense)
        if costs.ndim != 2:
            raise ValueError("`costs` must have shape (observations, objectives).")

        # It's a 2D ndarray -> copy it
        costs = costs.copy()

    elif user_has_package("pandas"):
        import pandas as pd

        if not isinstance(costs, pd.DataFrame):
            return validate_inputs(np.asarray(costs), sense=sense)
    else:
        return validate_inputs(np.asarray(costs), sense=sense)

    # if (not (isinstance(costs, np.ndarray) and costs.ndim == 2):
    #    raise TypeError("`costs` must be a NumPy array with 2 dimensions or pandas DataFrame.")

    n_costs, n_objectives = costs.shape

    if sense is None:
        return costs, ["min"] * n_objectives
    else:
        sense = list(sense)

    if not isinstance(sense, collections.abc.Sequence):
        raise TypeError("`sense` parameter must be a sequence (e.g. list).")

    if not len(sense) == n_objectives:
        raise ValueError("Length of `sense` must match second dimensions (i.e. columns).")

    # Convert functions "min" and "max" to their names
    sense = [s.__name__ if callable(s) else s for s in sense]
    if not all(isinstance(s, str) for s in sense):
        raise TypeError("`sense` parameter must be a sequence of strings.")
    sense = [s.lower() for s in sense]

    sense_map = {"min": "min", "minimum": "min", "max": "max", "maximum": "max", "diff": "diff", "different": "diff"}

    sense = [sense_map.get(s) for s in sense]

    # Verify that the strings are of correct format
    valid = ["min", "max", "diff"]
    if not all(s in valid for s in sense):
        raise TypeError("`sense` must be one of: {}".format(valid))

    return costs, sense


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules", "--maxfail=5", "--cache-clear", "--color", "yes", ""])
