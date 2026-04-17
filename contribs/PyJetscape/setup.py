"""Minimal setup.py shim — compile the C++ extension with CMake first.

  cmake -B build -DBUILD_PYJETSCAPE=ON -DJETSCAPE_DIR=/path/to/X-SCAPE/build
  cmake --build build

Then install in development mode:

  pip install -e contribs/PyJetscape
"""
from setuptools import setup
setup()
