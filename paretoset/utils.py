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
    """Sanitize user inputs for the `paretoset` function."""

    # The input is an np.ndarray
    if isinstance(costs, np.ndarray):
        if costs.ndim != 2:
            raise ValueError("`costs` must have shape (observations, objectives).")
    elif user_has_package("pandas"):
        import pandas as pd

        if isinstance(costs, pd.DataFrame):
            costs = costs.to_numpy()
        else:
            raise TypeError("`costs` must be a NumPy array with 2 dimensions or pandas DataFrame.")
    else:
        raise TypeError("`costs` must be a NumPy array with 2 dimensions or pandas DataFrame.")

    assert isinstance(costs, np.ndarray)

    if sense is None:
        return costs, sense

    n_costs, n_objectives = costs.shape

    if not isinstance(sense, collections.abc.Sequence):
        raise TypeError("`sense` parameter must be a sequence (e.g. list).")

    if not len(sense) == n_objectives:
        raise ValueError("Length of `sense` must match second dimensions (i.e. columns).")

    # Convert functions "min" and "max" to their names
    sense = [s.__name__ if callable(s) else s for s in sense]
    if not all(isinstance(s, str) for s in sense):
        raise TypeError("`sense` parameter must be a sequence of strings.")
    sense = [s.lower() for s in sense]

    # Verify that the strings are of correct format
    valid = ["min", "minimum", "max", "maximum", "diff", "difference"]
    if not all(s in valid for s in sense):
        raise TypeError("`sense` must be one of: {}".format(valid))

    return costs, sense
