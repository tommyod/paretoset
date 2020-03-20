#!/bin/sh
pip install twine;
python setup.py sdist bdist_wheel;
python -m twine upload dist/* -u tommyod -p $TWINE --skip-existing;