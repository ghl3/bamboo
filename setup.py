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
    name="pandas-bamboo",
    packages=["bamboo"],
    version='0.1.5',
    author="George Herbert Lewis",
    author_email="ghl227@gmail.com",
    license='The MIT License: http://www.opensource.org/licenses/mit-license.php',
    description="Data manipulation for python using Pandas",
    url = 'https://github.com/ghl3/bamboo',
    download_url = 'https://github.com/ghl3/bamboo/tarball/0.1.2',
    install_requires=[
        'pandas', 'numpy', 'matplotlib', 'singledispatch'
    ],
)
