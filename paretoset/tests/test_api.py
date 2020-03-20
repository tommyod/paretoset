from paretoset import paretoset
import numpy as np


import pytest
import pandas as pd


import pytest
import numpy as np
import itertools


allowed_min_values = [min, "min", "MIN", "minimum", "Min"]
allowed_max_values = [max, "max", "MAX", "maximum", "Max"]
bools = [True, False]


class TestReadmeExamples:
    def test_example_hotels(self):
        """Example in the readme."""
        from paretoset import paretoset
        import pandas as pd

        hotels = pd.DataFrame(
            {"price": [50, 53, 62, 87, 83, 39, 60, 44], "distance_to_beach": [13, 21, 19, 13, 5, 22, 22, 25]}
        )
        mask = paretoset(hotels, sense=["min", "min"])
        effi_hotels = hotels[mask]

    def test_example_salespeople(self):
        """Example in the readme."""
        from paretoset import paretoset
        import pandas as pd

        salespeople = pd.DataFrame(
            {
                "salary": [94, 107, 67, 87, 153, 62, 43, 115, 78, 77, 119, 127],
                "sales": [123, 72, 80, 40, 64, 104, 106, 135, 61, 81, 162, 60],
                "department": ["c", "c", "c", "b", "b", "a", "a", "c", "b", "a", "b", "a"],
            }
        )
        mask = paretoset(salespeople, sense=["min", "max", "diff"])
        top_performers = salespeople[mask]

    def test_example_optimization(self):
        """Example in the readme."""
        from paretoset import paretoset
        import numpy as np
        from collections import namedtuple

        np.random.seed(42)

        # Create Solution objects holding the problem solution and objective values
        Solution = namedtuple("Solution", ["solution", "objective_values"])
        solutions = [Solution(solution=object, objective_values=np.random.randn(2)) for _ in range(999)]

        # Create an array of shape (solutions, objectives) and compute the non-dominated set
        objective_values_array = np.vstack([s.objective_values for s in solutions])
        mask = paretoset(objective_values_array, sense=[min, min])

        # Filter the list of solutions, keeping only the non-dominated solutions
        efficient_solutions = [solution for (solution, m) in zip(solutions, mask) if m]

    def test_example_several_different(self):
        """A small example worked by hand."""

        df = pd.DataFrame(
            {
                "col1": ["a", "a", "a", "a", "a", "b", "b", "b", "b"],
                "col2": ["A", "A", "B", "B", "B", "A", "B", "B", "B"],
                "col3": [-1, 0, 0, 1, 1, -2, 1, 2, 1],
                "col4": [-1, 1, 1, 1, 0, -2, 0, 0, 0],
            }
        )
        sense = ["diff", "diff", "min", "min"]
        mask = paretoset(df, sense=sense, distinct=True)
        expected = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0], dtype=np.bool_)
        assert np.all(mask == expected)

        mask = paretoset(df, sense=sense, distinct=False)
        expected = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1], dtype=np.bool_)
        assert np.all(mask == expected)

    @pytest.mark.parametrize("operator, distinct", list(itertools.product(["min", "max"], bools)))
    def test_on_non_numeric_data(self, operator, distinct):
        """Test that non-numeric data fails."""

        df = pd.DataFrame({"col1": [0, 1, 0, 1], "col2": ["A", "A", "B", "B"]})

        # This will not work because 'operator' is applied over non-numeric data
        with pytest.raises(TypeError):
            paretoset(df, sense=["diff", operator], distinct=distinct)

        # Same as above
        with pytest.raises(TypeError):
            paretoset(df, sense=[operator, operator], distinct=distinct)
            
        # Fails on NumPy arrays too
        with pytest.raises(TypeError):
            data = np.array([[1, 1], [0, 1]], dtype=object)
            paretoset(data, sense=[operator, operator], distinct=distinct)

        # This will work, because the operator works on numeric data
        mask = paretoset(df, sense=[operator, "diff"], distinct=distinct)
        assert np.any(mask)

    @pytest.mark.parametrize("dtype", [np.int_, np.float_])
    def test_diff_on_numeric_data(self, dtype):
        """Test that 'diff' works on numeric data."""
        # Generate random data
        np.random.seed(42)
        costs = np.random.randint(low=-2, high=2, size=(99, 2))
        costs = np.asarray(costs, dtype=dtype)

        assert np.any(paretoset(costs, sense=["diff", "min"]))


class TestParetoSetAPI:
    @pytest.mark.parametrize(
        "dtype, distinct, use_numba", list(itertools.product([np.asarray, pd.DataFrame], bools, bools))
    )
    def test_at_least_one_chosen(self, dtype, distinct, use_numba):
        """Test that the input is not mutated."""

        # Generate random data
        np.random.seed(42)
        costs = np.random.randint(low=-2, high=2, size=(99, 3))

        # Get the mask
        mask = paretoset(dtype(costs), sense=[min, max, min], distinct=distinct, use_numba=use_numba)

        # Verify that at least one element is chosen
        assert np.any(mask)

    @pytest.mark.parametrize(
        "dtype, distinct, use_numba", list(itertools.product([np.asarray, pd.DataFrame], bools, bools))
    )
    def test_no_side_effects(self, dtype, distinct, use_numba):
        """Test that the input is not mutated."""

        # Generate random data
        np.random.seed(42)
        costs = np.random.randint(low=-2, high=2, size=(99, 3))
        costs = dtype(costs)

        # Copy the data before passing into function
        costs_before = costs.copy()
        paretoset(costs, sense=[min, max, min], distinct=distinct, use_numba=use_numba)

        # The input argument is not changed/mutated in any way
        assert costs_before.shape == costs.shape
        assert np.all(costs_before == costs)

    @pytest.mark.parametrize("sense_min, sense_max", list(itertools.product(allowed_min_values, allowed_max_values)))
    def test_sense_argument(self, sense_min, sense_max):
        """Test the synonyms for the `sense` argument."""

        np.random.seed(42)
        costs = np.random.randint(low=-2, high=2, size=(99, 3))

        mask1 = paretoset(costs, sense=["min", "max", "diff"])
        mask2 = paretoset(costs, sense=[sense_min, sense_max, "diff"])

        # Masks are equal
        assert np.all(mask1 == mask2)


class TestPandasInputs:
    def test_pandas_no_args(self):
        costs = np.random.RandomState(123).randn(5, 2)
        df = pd.DataFrame(costs, columns=["a", "b"])
        df_before = df.copy()
        mask = paretoset(df)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool
        assert (df == df_before).all().all()

    def test_pandas_with_args(self):
        costs = np.random.RandomState(123).randn(5, 2)
        df = pd.DataFrame(costs, columns=["a", "b"])
        df_before = df.copy()
        mask = paretoset(df, sense=[min, max])

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool
        assert (df == df_before).all().all()

    def test_pandas_with_diff_arg(self):
        costs = np.random.RandomState(123).randn(24, 2)
        df = pd.DataFrame(costs, columns=["a", "b"])
        df["department"] = ["a", "b", "c"] * 8
        df_before = df.copy()
        mask = paretoset(df, sense=["min", "max", "diff"])

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool
        assert (df == df_before).all().all()


if __name__ == "__main__":
    pytest.main(args=[__file__, "--doctest-modules", "--maxfail=5", "-v", "--cache-clear", "--color", "yes", ""])

    import pandas as pd

    df = pd.DataFrame(
        {
            "col1": ["a", "a", "a", "a", "a", "b", "b", "b", "b"],
            "col2": ["A", "A", "B", "B", "B", "A", "B", "B", "B"],
            "col3": [-1, 0, 0, 1, 1, -2, 1, 2, 1],
            "col4": [-1, 1, 1, 1, 0, -2, 0, 0, 0],
        }
    )
    mask = paretoset(df, sense=["diff", "diff", "min", "min"], distinct=True)
    print(mask)
    expected = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0], dtype=np.bool_)
