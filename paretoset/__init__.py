#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the Apriori algorithm.
"""

# We use semantic versioning
# See: https://semver.org/
__version__ = "1.0.0"

import sys
from paretoset.paretoset import paretoset


def run_tests():
    """
    Run all tests.
    """
    import pytest
    import os

    base, _ = os.path.split(__file__)
    pytest.main(args=[base, "--doctest-modules"])


if (sys.version_info[0] < 3) or (sys.version_info[1] < 6):
    msg = "The `paretoset` package only works for Python 3.6+."
    raise Exception(msg)
