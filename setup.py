#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

setup(
    name="bamboo",
    version='0.1',
    author="George Herbert Lewis",
    author_email="ghl227@gmail.com",
    packages=["bamboo"],
    description="Data manipulation for python using Pandas",
)
