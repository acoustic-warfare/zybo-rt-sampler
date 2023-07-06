from setuptools import setup
from Cython.Build import cythonize

import numpy


setup (
    name = 'api',
    ext_modules = cythonize(["src/*.pyx"], build_dir="build"),
    include_dirs=[numpy.get_include()],
)