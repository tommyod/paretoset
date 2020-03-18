

from skyline import skyline
import numpy as np


import pytest
import pandas as pd


import pytest
import numpy as np
import itertools


allowed_min_values = [min, "min", "MIN", "minimum", "Min"]
allowed_max_values = [max, "max", "MAX", "maximum", "Max"]



class TestReadmeExamples:

    def test_example_hotels(self):
        """Example in the readme."""
        from skyline import skyline
        import pandas as pd

        hotels = pd.DataFrame(
            {"price": [50, 53, 62, 87, 83, 39, 60, 44], "distance_to_beach": [13, 21, 19, 13, 5, 22, 22, 25]})
        mask = skyline(hotels, sense=["min", "min"])
        skyline_hotels = hotels[mask]

    def test_example_salespeople(self):
        """Example in the readme."""
        from skyline import skyline
        import pandas as pd

        salespeople = pd.DataFrame(
            {
                "salary": [94, 107, 67, 87, 153, 62, 43, 115, 78, 77, 119, 127],
                "sales": [123, 72, 80, 40, 64, 104, 106, 135, 61, 81, 162, 60],
                "department": ["c", "c", "c", "b", "b", "a", "a", "c", "b", "a", "b", "a"],
            }
        )
        mask = skyline(salespeople, sense=["min", "max", "diff"])
        top_performers = salespeople[mask]

    def test_example_optimization(self):
        """Example in the readme."""
        from skyline import skyline
        import numpy as np
        from collections import namedtuple

        np.random.seed(42)

        # Create Solution objects holding the problem solution and objective values
        Solution = namedtuple("Solution", ["solution", "objective_values"])
        solutions = [Solution(solution=object, objective_values=np.random.randn(2)) for _ in range(999)]

        # Create an array of shape (solutions, objectives) and compute the non-dominated set
        objective_values_array = np.vstack([s.objective_values for s in solutions])
        mask = skyline(objective_values_array, sense=[min, min])

        # Filter the list of solutions, keeping only the non-dominated solutions
        efficient_solutions = [solution for (solution, m) in zip(solutions, mask) if m]




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

    @pytest.mark.parametrize("sense_min, sense_max", list(itertools.product(allowed_min_values, allowed_max_values)))
    def test_sense_argument(self, sense_min, sense_max):

        costs = np.random.RandomState(123).randn(10, 3)
        costs_before = costs.copy()

        mask1 = skyline(costs, sense=["min", "max", "diff"])
        mask2 = skyline(costs, sense=[sense_min, sense_max, "diff"])
        assert np.all(mask1 == mask2)
        assert np.allclose(costs, costs_before)


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

