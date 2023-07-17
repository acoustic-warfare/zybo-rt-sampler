from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy
import glob

c_files = []
c_files.extend(glob.glob("src/*.c"))
c_files.extend(glob.glob("src/antenna/*.c"))

setup (
    name = 'Module',
    ext_modules = cythonize(
        [   #Extension("VideoPlayer", ["src/modules/VideoPlayer.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            #Extension("test", ["src/test.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            #Extension("foo", ["src/foo.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            Extension("microphone_array", ["src/microphone_array.pyx"] + c_files, include_dirs=["src/"],
                        extra_compile_args = ["-O3", "-march=native", "-mavx2", "-lm"]),
            Extension("VideoPlayer", ["src/modules/VideoPlayer.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            #Extension("TruncAndSum", ["src/modules/TruncAndSum.pyx"] + c_files, include_dirs=["src/"],
            #            extra_compile_args = ["-O3", "-march=native", "-mavx2", "-lm"]),
            #Extension("Beamformer", ["src/modules/Beamformer.pyx"] + c_files, include_dirs=["src/"],
            #            extra_compile_args = ["-O3", "-march=native", "-mavx2", "-lm"]),
            Extension("directions", ["src/directions.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
        ],

        build_dir="build"
    ),
    include_dirs=[numpy.get_include()],
)
