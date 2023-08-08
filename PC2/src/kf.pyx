# cython: language_level=3
# distutils: language=c++

import numpy as np
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()

from libcpp.vector cimport vector

cdef extern from "kf.hpp":
    cdef cppclass KalmanFilter3D:
        void updatef(const vector[float] &measurment)
        vector[float] getStatef()
        vector[float] predictf(int N)

cdef class CyKF:
    cdef KalmanFilter3D* _cpp_obj
    cdef public vector[float] state, prediction, vec

    def __cinit__(self):
        self._cpp_obj = new KalmanFilter3D()

        for i in range(3):
            self.state.push_back(0.0)
            self.prediction.push_back(0.0)
            self.vec.push_back(0.0)

    def __dealloc__(self):
        del self._cpp_obj

    def get_state(self):
        self.state = self._cpp_obj.getStatef()
        return np.array(self.state, dtype=np.float32)

    def predict(self, n: int):
        self.prediction = self._cpp_obj.predictf(n)
        return np.array(self.prediction, dtype=np.float32)

    def update(self, meas):
        self.vec.clear()
        self.vec.push_back(meas[0])
        self.vec.push_back(meas[1])
        self.vec.push_back(meas[2])
        self._cpp_obj.updatef(self.vec)