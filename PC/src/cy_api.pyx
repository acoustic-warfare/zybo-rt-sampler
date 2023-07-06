# cython: language_level=3
# distutils: language=c

# File name: cy_api.pyx
# Description: Cython Module With Added NumPy Support For beamformer
# Author: Irreq
# Date: 2022-12-29

# Main API for C functions

from .utils import test

def entrypoint():
    print("Working")
    test()