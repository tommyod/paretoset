import numpy as np

from paretoset.algorithms_numpy import paretoset_efficient
from paretoset.utils import user_has_package, validate_inputs

if user_has_package("pandas"):
    import pandas as pd

if user_has_package("numba"):
    from paretoset.algorithms_numba import paretoset_jit


def paretoset(costs, sense=None, distinct=True, use_numba=True):
    """Return mask indicating the Pareto set of a NumPy array or pandas DataFrame.


    Parameters
    ----------
    costs : np.ndarray or pd.DataFrame
        Array or DataFrame of shape (observations, objectives).
    sense : collection
        List or tuple with `min` (default), `max` or `diff`. If None, minimization is assumed
        for every objective (column). The parameter `diff` indicates that each observation must
        be different for that objective.
    distinct : bool
        How to treat duplicates. If True, only the first one is returned.
        If False, every identical observation is returned.
    use_numba : bool
        If True, numba will be used if it is installed.

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
        paretoset_algorithm = paretoset_jit
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
    if user_has_package("pandas"):
        import pandas as pd

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
        if user_has_package("pandas"):
            import pandas as pd

            if isinstance(costs, pd.DataFrame):
                costs = costs.to_numpy(copy=True)

        assert isinstance(costs, np.ndarray)
        return paretoset_algorithm(costs, distinct=distinct)

    n_costs, n_objectives = costs.shape

    if not diff_cols:
        # Its an array
        if not isinstance(costs, np.ndarray):
            costs = costs.to_numpy(copy=True)
        for col in max_cols:
            costs[:, col] = -costs[:, col]

        return paretoset_algorithm(costs, distinct=distinct)

    if diff_cols and not user_has_package("pandas"):
        raise ModuleNotFoundError("The `diff` sense requires pandas. See: https://pandas.pydata.org/")

    if user_has_package("pandas"):
        import pandas as pd

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
    """Return mask indicating the Pareto set of a NumPy array or pandas DataFrame.


    Parameters
    ----------
    costs : np.ndarray or pd.DataFrame
        Array or DataFrame of shape (observations, objectives).
    sense : collection
        List or tuple with `min` (default), `max` or `diff`. If None, minimization is assumed
        for every objective (column). The parameter `diff` indicates that each observation must
        be different for that objective.
    distinct : bool
        How to treat duplicates. If True, only the first one is returned.
        If False, every identical observation is returned.
    use_numba : bool
        If True, numba will be used if it is installed.

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
        paretorank_algorithm = pareto_rank_naive
    else:
        paretorank_algorithm = pareto_rank_naive

    costs, sense = validate_inputs(costs=costs, sense=sense)

    # TODO: INF and NaN

    if sense is None:
        return paretorank_algorithm(costs, distinct=distinct)

    n_costs, n_objectives = costs.shape

    diff_cols = [i for i in range(n_objectives) if sense[i] == "diff"]
    max_cols = [i for i in range(n_objectives) if sense[i] == "max"]
    min_cols = [i for i in range(n_objectives) if sense[i] == "min"]

    if diff_cols and not user_has_package("pandas"):
        raise ModuleNotFoundError("The `diff` sense requires pandas. See: https://pandas.pydata.org/")

    for col in max_cols:
        costs[:, col] = -costs[:, col]

    if not diff_cols:
        return paretorank_algorithm(costs, distinct=distinct)

    df = pd.DataFrame(costs)

    # If `object` columns are present and they can be converted, do it.
    for col in df.columns:
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
        ranks = paretorank_algorithm(relevant_data.copy(), distinct=distinct)

        # The `pd.DataFrame.groupby.indices` dict holds the row indices of the group
        data_mask = groupby.indices[key]
        all_ranks[data_mask] = ranks

    return all_ranks


if __name__ == "__main__":
    import pytest

    pytest.main(args=[".", "--doctest-modules", "--maxfail=5", "--cache-clear", "--color", "yes", ""])
