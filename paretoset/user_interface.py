import numpy as np

from paretoset.algorithms_numpy import paretoset_efficient, pareto_rank_naive
from paretoset.utils import user_has_package, validate_inputs


import pandas as pd

if user_has_package("numba"):
    from paretoset.algorithms_numba import BNL


def paretoset(costs, sense=None, distinct=True, use_numba=True):
    """Return boolean mask indicating the Pareto set of (non-NaN) numerical data.

    The input data in `costs` can be either a pandas DataFrame or a NumPy ndarray
    of shape (observations, objectives). The user is responsible for dealing with
    NaN values *before* sending data to this function. Only numerical data is
    allowed, with the exception of `diff` (different) columns.

    Parameters
    ----------
    costs : np.ndarray or pd.DataFrame
        Array or DataFrame of shape (observations, objectives).
    sense : list
        List with strings for each column (objective). The value `min` (default)
        indicates minimization, `max` indicates maximization and `diff` indicates
        different values. Using `diff` is equivalent to a group-by operation
        over the columns marked with `diff`. If None, minimization is assumed.
    distinct : bool
        How to treat duplicate rows. If `True`, only the first duplicate is returned.
        If `False`, every identical observation is returned instead.
    use_numba : bool
        If True, numba will be used if it is installed by the user.

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

    The `distinct` parameter:

    >>> paretoset([0, 0], distinct=True)
    array([ True, False])
    >>> paretoset([0, 0], distinct=False)
    array([ True,  True])
    """

    if user_has_package("numba") and use_numba:
        paretoset_algorithm = BNL
    else:
        paretoset_algorithm = paretoset_efficient

    costs, sense = validate_inputs(costs=costs, sense=sense)
    assert isinstance(sense, list)

    n_costs, n_objectives = costs.shape

    diff_cols = [i for i in range(n_objectives) if sense[i] == "diff"]
    max_cols = [i for i in range(n_objectives) if sense[i] == "max"]
    min_cols = [i for i in range(n_objectives) if sense[i] == "min"]

    # Check data types (MIN and MAX must be numerical)
    message = "Data must be numerical. Please convert it. Data has type: {}"

    if isinstance(costs, pd.DataFrame):
        data_types = [costs.dtypes.values[i] for i in (max_cols + min_cols)]
        if any(d == np.dtype("O") for d in data_types):
            raise TypeError(message.format(data_types))

    else:
        if costs.dtype == np.dtype("O"):
            raise TypeError(message.format(costs.dtype))

    # CASE 1: THE ONLY SENSE IS MINIMIZATION
    # ---------------------------------------
    if all(s == "min" for s in sense):
        if isinstance(costs, pd.DataFrame):
            costs = costs.to_numpy(copy=True)
        return paretoset_algorithm(costs, distinct=distinct)

    n_costs, n_objectives = costs.shape

    if not diff_cols:
        # Its an array
        if not isinstance(costs, np.ndarray):
            costs = costs.to_numpy(copy=True)
        for col in max_cols:
            costs[:, col] = -costs[:, col]

        return paretoset_algorithm(costs, distinct=distinct)

    if isinstance(costs, pd.DataFrame):
        df = costs.copy()  # Copy to avoid mutating inputs
        df.columns = np.arange(n_objectives)
    else:
        df = pd.DataFrame(costs)

    assert isinstance(df, pd.DataFrame)
    assert np.all(df.columns == np.arange(n_objectives))

    # If `object` columns are present and they can be converted, do it.
    for col in max_cols + min_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    is_efficient = np.zeros(n_costs, dtype=np.bool_)

    # Create the groupby object
    # We could've implemented our own groupby, but choose to use pandas since
    # it's likely better than what we can come up with on our own.
    groupby = df.groupby(diff_cols)

    # Iteration through the groups
    for key, data in groupby:

        # Get the relevant data for the group and compute the efficient points
        relevant_data = data[max_cols + min_cols].to_numpy(copy=True)
        efficient_mask = paretoset_algorithm(relevant_data.copy(), distinct=distinct)

        # The `pd.DataFrame.groupby.indices` dict holds the row indices of the group
        data_mask = groupby.indices[key]
        is_efficient[data_mask] = efficient_mask

    return is_efficient


def paretorank(costs, sense=None, distinct=True, use_numba=True):
    """Return integer array with Pareto ranks of (non-NaN) numerical data.

    Observations in the Pareto set are assigned rank 1. After removing the Pareto
    set, the Pareto set of the remaining data is assigned rank 2, and so forth.

    The input data in `costs` can be either a pandas DataFrame or a NumPy ndarray
    of shape (observations, objectives). The user is responsible for dealing with
    NaN values *before* sending data to this function. Only numerical data is
    allowed, with the exception of `diff` (different) columns.

    Parameters
    ----------
    costs : np.ndarray or pd.DataFrame
        Array or DataFrame of shape (observations, objectives).
    sense : list
        List with strings for each column (objective). The value `min` (default)
        indicates minimization, `max` indicates maximization and `diff` indicates
        different values. Using `diff` is equivalent to a group-by operation
        over the columns marked with `diff`. If None, minimization is assumed.
    distinct : bool
        How to treat duplicate rows. If `True`, only the first duplicate is returned.
        If `False`, every identical observation is returned instead.
    use_numba : bool
        If True, numba will be used if it is installed by the user.

    Returns
    -------
    ranks : np.ndarray
        Integer array with Pareto ranks of the observations.

    Examples
    --------
    >>> from paretoset import paretoset
    >>> import numpy as np
    >>> costs = np.array([[2, 0], [1, 1], [0, 2], [3, 3]])
    >>> paretorank(costs)
    array([1, 1, 1, 2])
    >>> paretorank(costs, sense=["min", "max"])
    array([3, 2, 1, 1])

    The `distinct` parameter:

    >>> paretorank([0, 0], distinct=True)
    array([1, 2])
    >>> paretorank([0, 0], distinct=False)
    array([1, 1])
    """

    if user_has_package("numba") and use_numba:
        paretorank_algorithm = pareto_rank_naive
    else:
        paretorank_algorithm = pareto_rank_naive

    costs, sense = validate_inputs(costs=costs, sense=sense)
    assert isinstance(sense, list)

    n_costs, n_objectives = costs.shape

    diff_cols = [i for i in range(n_objectives) if sense[i] == "diff"]
    max_cols = [i for i in range(n_objectives) if sense[i] == "max"]
    min_cols = [i for i in range(n_objectives) if sense[i] == "min"]

    # Check data types (MIN and MAX must be numerical)
    message = "Data must be numerical. Please convert it. Data has type: {}"

    if isinstance(costs, pd.DataFrame):
        data_types = [costs.dtypes.values[i] for i in (max_cols + min_cols)]
        if any(d == np.dtype("O") for d in data_types):
            raise TypeError(message.format(data_types))

    else:
        if costs.dtype == np.dtype("O"):
            raise TypeError(message.format(costs.dtype))

    # CASE 1: THE ONLY SENSE IS MINIMIZATION
    # ---------------------------------------
    if all(s == "min" for s in sense):
        if isinstance(costs, pd.DataFrame):
            costs = costs.to_numpy(copy=True)
        return paretorank_algorithm(costs, distinct=distinct, use_numba=use_numba)

    n_costs, n_objectives = costs.shape

    if not diff_cols:
        # Its an array
        if not isinstance(costs, np.ndarray):
            costs = costs.to_numpy(copy=True)
        for col in max_cols:
            costs[:, col] = -costs[:, col]

        return paretorank_algorithm(costs, distinct=distinct, use_numba=use_numba)

    if isinstance(costs, pd.DataFrame):
        df = costs.copy()  # Copy to avoid mutating inputs
        df.columns = np.arange(n_objectives)
    else:
        df = pd.DataFrame(costs)

    assert isinstance(df, pd.DataFrame)
    assert np.all(df.columns == np.arange(n_objectives))

    # If `object` columns are present and they can be converted, do it.
    for col in max_cols + min_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    all_ranks = np.zeros(n_costs, dtype=np.int_)

    # Create the groupby object
    # We could've implemented our own groupby, but choose to use pandas since
    # it's likely better than what we can come up with on our own.
    groupby = df.groupby(diff_cols)

    # Iteration through the groups
    for key, data in groupby:

        # Get the relevant data for the group and compute the efficient points
        relevant_data = data[max_cols + min_cols].to_numpy(copy=True)
        ranks = paretorank_algorithm(relevant_data.copy(), distinct=distinct, use_numba=use_numba)

        # The `pd.DataFrame.groupby.indices` dict holds the row indices of the group
        data_mask = groupby.indices[key]
        all_ranks[data_mask] = ranks

    return all_ranks


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules", "--maxfail=5", "--cache-clear", "--color", "yes", ""])
