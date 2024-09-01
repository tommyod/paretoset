#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the Pareto (non-dominated) set, i.e., skyline operator/query.
"""

# We use semantic versioning
# See: https://semver.org/
__version__ = "1.2.4"
__author__ = "tommyod"

from paretoset.user_interface import paretoset, paretorank
from paretoset.algorithms_numpy import crowding_distance


def run_tests():
    """
    Run all tests.
    """
    import pytest
    import os

    base, _ = os.path.split(__file__)
    pytest.main(args=[base, "--doctest-modules"])
