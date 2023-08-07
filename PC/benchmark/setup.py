from setuptools import setup, Extension
from Cython.Build import cythonize

# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True


import numpy
import glob

c_files = []
c_files.extend(glob.glob("src/*.c"))
c_files.extend(glob.glob("src/algorithms/*.c"))

CFLAGS = "-finline-functions -O3 -march=native -mavx2 -lm -lrt -lasound -ljack -lpthread -lportaudio"
CFLAGS = CFLAGS.split(" ")

setup (
    name = 'Module',
    ext_modules = cythonize(
        [
            Extension("beamformer", ["src/main.pyx"] + c_files, 
                      include_dirs=["src/"], extra_compile_args = CFLAGS, libraries=['portaudio']),
            Extension("directions", ["src/directions.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            Extension("tests", ["src/benchmark.pyx"], include_dirs=["src/"], extra_compile_args = CFLAGS),
            #Extension("visual", ["src/visual.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            Extension("kf", ["src/kf.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            # Extension("mic", ["src/mic.pyx"]+c_files, include_dirs=["src/"], extra_compile_args = CFLAGS, libraries=['portaudio']),
        ],

        build_dir="build",
        # annotate = True
    ),
    include_dirs=[numpy.get_include()],
    python_requires='>=3',
)
