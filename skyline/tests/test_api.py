

from skyline import skyline
import numpy as np


import pytest
import pandas as pd


class TestReadmeExamples:

    def test_example_1(self):
        """First example in the readme."""
        prices =




class TestNumPyInputs:


    def test_numpy_no_args(self):
        costs = np.random.RandomState(123).randn(5, 2)
        costs_before = costs.copy()
        mask = skyline(costs)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool
        assert np.allclose(costs_before, costs)

    def test_numpy_with_args(self):
        costs = np.random.RandomState(123).randn(5, 2)
        costs_before = costs.copy()
        mask = skyline(costs, sense=["min", "max"])

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool
        assert np.allclose(costs_before, costs)


class TestPandasInputs:


    def test_pandas_no_args(self):
        costs = np.random.RandomState(123).randn(5, 2)
        df = pd.DataFrame(costs, columns=["a", "b"])
        df_before = df.copy()
        mask = skyline(df)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool
        assert (df == df_before).all().all()


    def test_pandas_with_args(self):
        costs = np.random.RandomState(123).randn(5, 2)
        df = pd.DataFrame(costs, columns=["a", "b"])
        df_before = df.copy()
        mask = skyline(df, sense=[min, max])

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool
        assert (df == df_before).all().all()

    def test_pandas_with_diff_arg(self):
        costs = np.random.RandomState(123).randn(24, 2)
        df = pd.DataFrame(costs, columns=["a", "b"])
        df["department"] = ["a", "b", "c"] * 8
        df_before = df.copy()
        mask = skyline(df, sense=["min", "max", "diff"])

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool
        assert (df == df_before).all().all()

