#!/usr/bin/env python3

# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD !

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages = ['simrad_raw_radar'], # octantis2_driver/src/octantis2_driver
    package_dir = {'': 'src'},
)

setup(**setup_args)
