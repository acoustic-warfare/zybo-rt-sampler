CC = gcc
CYCC = cython3

PYTHON_PATH = `python3 -c "import distutils.sysconfig;print(distutils.sysconfig.get_python_inc())"`

CFLAGS = -O3 -march=native -mavx2
LDFLAGS = -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing

.PHONY: all

all: lib/beamformer.so

config:
	python3 src/build_config.py

lib/beamformer.so: config
	python3 setup.py build_ext --inplace --build-lib=build
	mv *.so lib/
	ln src/visual.py lib/visual.py
	cp lib/*.so ../application/lib/	

# lib/beamformer.so1: config
# 	cp -r src/* build/
# 	$(CYCC) src/main.pyx -o build/main.c
# 	$(CC) $(CFLAGS) $(LDFLAGS) -lm -I $(PYTHON_PATH) -o lib/beamformer.so build/main.c


test: lib/beamformer.so
	python3 test.py

fig: clean lib/beamformer.so
	python3 exe.py

listen: config
	gcc -Wall -lm -o run src/*.c src/algorithms/*.c -O3 -march=native -mavx2 -lm -lrt -lasound -ljack -lpthread -lportaudio -lm

.PHONY: clean

clean:
	echo "Removing Generated Build Files"
	ls build/ | grep -xv ".gitignore" | xargs -I {} sh -c "rm -r build/{}"
	ls lib/ | grep -xv ".gitignore" | xargs -I {} sh -c "rm -r lib/{}"
	ls ../application/lib/ | grep -xv ".gitignore" | xargs -I {} sh -c "rm -r ../application/lib/{}"
