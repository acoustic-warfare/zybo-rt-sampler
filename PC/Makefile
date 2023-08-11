CC = gcc
CYCC = cython3

PYTHON_PATH = `python3 -c "import distutils.sysconfig;print(distutils.sysconfig.get_python_inc())"`

CFLAGS = -O3 -march=native -mavx2
LDFLAGS = -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing

.PHONY: all

all: lib/beamformer.so

config:
	python3 src/build_config.py
	cp -u -n src/config.h src/algorithms/config.h

lib/beamformer.so: config
	python3 setup.py build_ext --inplace --build-lib=build
	mkdir -p lib
	mv *.so lib/
	# cp unused_mics.npy lib/
	cp -u -n src/visual.py lib/visual.py
	cp -u -n lib/* application/lib/
	cp -u -n interface/config.py application/interface/config.py
	
bor: config
	python3 setup.py build_ext --inplace --build-lib=build
	mv *.so lib/
	# cp lib/*.so ../application/lib/

yum: config	lib/beamformer.so
	python3 -c "from lib.mic import *;start()"

sym: config
	gcc -Wall -lm -o run playback.c src/*.c src/algorithms/*.c -O3 -march=native -mavx2 -lm -lrt -lasound -ljack -lpthread -lportaudio -lm
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

cytest: config
	cp -r src/* build/
	
	$(CC) -O3 --shared -fPIC -march=native -mavx2 -lm build/*.c build/algorithms/*.c -o build/delay.so -march=native -mavx2 -lm -lrt -lasound -ljack -lpthread -lportaudio -lm
	
	$(CYCC) src/mic.pyx -o build/mic.c
	$(CC) -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -lm -march=native -mavx2 -I $(PYTHON_PATH) -o lib/mic.so build/mic.c build/delay.so -march=native -mavx2 -lm -lrt -lasound -ljack -lpthread -lportaudio -lm


.PHONY: clean

clean:
	echo "Removing Generated Build Files"
	ls build/ | grep -xv ".gitignore" | xargs -I {} sh -c "rm -r build/{}"
	ls lib/ | grep -xv ".gitignore" | xargs -I {} sh -c "rm -r lib/{}"
	# ls ../application/lib/ | grep -xv ".gitignore" | xargs -I {} sh -c "rm -r ../application/lib/{}"
	# ls ../application/interface/ | grep -xv ".gitignore" | xargs -I {} sh -c "rm -r ../application/interface/{}"
