import numpy as np

from paretoset.algorithms_numpy import paretoset_efficient

import collections.abc

try:
    import pandas as pd

    USER_HAS_PANDAS = True
except ImportError:
    USER_HAS_PANDAS = False

try:
    import numba

    USER_HAS_NUMBA = True
except ImportError:
    USER_HAS_NUMBA = False


def paretoset(costs, sense=None):

    # The input is an np.ndarray
    if isinstance(costs, np.ndarray):
        if costs.ndim != 2:
            raise ValueError("Array must have shape (observations, objectives).")
    elif USER_HAS_PANDAS and isinstance(costs, pd.DataFrame):
        costs = costs.to_numpy()
    else:
        TypeError("`costs` must be a NumPy ndarray or pandas DataFrame.")

    # TODO: INF and NaN

    if sense is None:
        return paretoset_efficient(costs.copy())

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

    # Take a copy since we will be changing the object
    costs = costs.copy()

    diff_cols = [i for i in range(n_objectives) if sense[i] in ("diff", "difference")]
    max_cols = [i for i in range(n_objectives) if sense[i] in ("max", "maximum")]
    min_cols = [i for i in range(n_objectives) if sense[i] in ("min", "minimum")]

    for col in max_cols:
        costs[:, col] = -costs[:, col]

    if not diff_cols:
        return paretoset_efficient(costs)

    df = pd.DataFrame(costs)
    is_efficient = np.zeros(n_costs, dtype=np.bool_)
    for key, data in df.groupby(diff_cols):
        data = data[max_cols + min_cols].to_numpy()
        mask = paretoset_efficient(data.copy())

        if not isinstance(key, tuple):
            insert_mask = df[diff_cols] == key
        else:
            insert_mask = (df[diff_cols] == key).all(axis=1)

        insert_mask = insert_mask.to_numpy().ravel()

        is_efficient[insert_mask] = mask

    return is_efficient
