import numpy as np

from paretoset.algorithms_numpy import paretoset_efficient
from paretoset.utils import user_has_package, validate_inputs

import collections.abc

if user_has_package("pandas"):
    import pandas as pd


def paretoset(costs, sense=None, use_numba=True):
    """Return mask indicating the Pareto set of a NumPy array or pandas DataFrame.


    Parameters
    ----------
    costs : np.ndarray or pd.DataFrame
        Array or DataFrame of shape (observations, objectives).
    sense : collection
        List or tuple with `min` (default), `max` or `diff`. If None, minimization is assumed
        for every objective (column). The parameter `diff` indicates that each observation must
        be different for that objective.
    use_numba : bool
        If True, numba will be used if it`s installed.

    Returns
    -------
    mask : np.ndarray
        Boolean mask with `True` for observations in the Pareto set.


    Examples
    --------
    >>> from paretoset import paretoset
    >>> import numpy as np
    >>> costs = np.array([[2, 0], [1, 1], [0, 2], [3, 3]])
    >>> paretoset(costs)
    array([ True,  True,  True, False])
    >>> paretoset(costs, sense=["min", "max"])
    array([False, False,  True,  True])
    """

    costs, sense = validate_inputs(costs=costs, sense=sense)

    # TODO: INF and NaN

    if sense is None:
        return paretoset_efficient(costs.copy())

    n_costs, n_objectives = costs.shape

    # Take a copy since we will be changing the object
    costs = costs.copy()

    diff_cols = [i for i in range(n_objectives) if sense[i] in ("diff", "difference")]
    max_cols = [i for i in range(n_objectives) if sense[i] in ("max", "maximum")]
    min_cols = [i for i in range(n_objectives) if sense[i] in ("min", "minimum")]

    if diff_cols and not user_has_package("pandas"):
        raise ModuleNotFoundError("The `diff` sense requires pandas. See: https://pandas.pydata.org/")

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
