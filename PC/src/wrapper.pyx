# cython: language_level=3
# distutils: language=c

import numpy as np
cimport numpy as np

import time

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()

import sys
sys.path.insert(0, "") # Access local modules located in . Enables 'from . import MODULE'

# Create specific data-types "ctypedef" assigns a corresponding compile-time type to DTYPE_t.
ctypedef np.float32_t DTYPE_t

# Constants
DTYPE_arr = np.float32

cdef extern from "beamformer.h":
    void foo(float *signal)
    void bar()
    int load(bool)


cdef void hello():
    cdef np.ndarray[DTYPE_t, ndim=3, mode = 'c'] arr
    data = [[[1, 2, 1],
            [3, 4, 5]]]
    arr = np.ascontiguousarray(np.array(data, dtype=DTYPE_arr))

    cdef DTYPE_t *arr_ptr

    arr_ptr = &arr[0, 0, 0]

    foo(arr_ptr)

    load(False)

    while True:
        print(3)
        time.sleep(1)

    print(arr)



def main():
    hello()
    bar()
    print("Hello world!")

