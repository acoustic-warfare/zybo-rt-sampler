from setuptools import setup, Extension
from Cython.Build import cythonize

# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True


import numpy
import glob

c_files = []
c_files.extend(glob.glob("src/*.c"))
# print(c_files)
c_files.extend(glob.glob("src/algorithms/*.c"))

CFLAGS = "-O3 -march=native -mavx2 -lm -lrt -lasound -ljack -lpthread -lportaudio -lm -Wall -O3 -march=native -mavx2 -lm -lrt -lasound -ljack -lpthread -lportaudio -lm"

setup (
    name = 'Module',
    ext_modules = cythonize(
        [
            Extension("beamformer", ["src/main.pyx"] + c_files, include_dirs=["src/"],
                        extra_compile_args = CFLAGS.split(" "), libraries=['portaudio']),
            #Extension("tests", ["src/benchmark.pyx"] + c_files, include_dirs=["src/"],
            #            extra_compile_args = CFLAGS.split(" ")),
            # Extension("VideoPlayer", ["src/modules/VideoPlayer.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            #Extension("TruncAndSum", ["src/modules/TruncAndSum.pyx"] + c_files, include_dirs=["src/"],
            #            extra_compile_args = ["-O3", "-march=native", "-mavx2", "-lm"]),
            #Extension("Beamformer", ["src/modules/Beamformer.pyx"] + c_files, include_dirs=["src/"],
            #            extra_compile_args = ["-O3", "-march=native", "-mavx2", "-lm"]),
            Extension("directions", ["src/directions.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            #Extension("visual", ["src/visual.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            Extension("kf", ["src/kf.pyx"], include_dirs=["src/"], extra_compile_args = ["-lm"]),
            # Extension("mic", ["src/mic.pyx"]+c_files, include_dirs=["src/"], extra_compile_args = CFLAGS.split(" "), libraries=['portaudio']),
            # Extension("mic", ["src/test/foo.pyx"]+glob.glob("src/test/*.c"), include_dirs=["src/test/"], extra_compile_args = CFLAGS.split(" "), libraries=['portaudio']),
        ],

        build_dir="build",
        # annotate = True
    ),
    include_dirs=[numpy.get_include()],
    python_requires='>=3',
)
