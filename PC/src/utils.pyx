# cython: language_level=3
# distutils: language=c

# File name: utils.pyx
# Description: Helper functions
# Author: Irreq

cdef extern from "config.h":
    cdef int N_MICROPHONES

n_microphones = N_MICROPHONES

def test():
    print(f"{__file__}")