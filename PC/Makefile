CC = gcc
CYCC = cython3

CFLAGS = `python3-config --cflags`
CFLAGS += -fPIE

# For x86_64 SIMD
CFLAGS += -Ofast -flto -ffast-math -march=native -mno-vzeroupper -msse3 -msse4 -mavx2 -mavx
# get only 3.XX
PYTHON_VERSION := $(python3 --version | grep -Po "\d+\.\d+" | echo)
PYTHON_MODULES = -I/usr/lib/python$(PYTHON_VERSION)/site-packages/numpy/core/include/

CFLAGS += -Ibuild/ -Isrc/ # Add build to path

CFLAGS += $(PYTHON_MODULES)
LFLAGS = `python3-config --embed --ldflags` # Must be --embed for >python3.8

LFLAGS += $(PYTHON_MODULES)
BIN = run

BUILD = build

.PHONY: all

all: lib/libsampler.so run

# build/cy_api.c:
# 	$(CYCC) -o build/cy_api.c src/cy_api.pyx
# 
# build/cy_api.o: build/cy_api.c
# 	$(CC) $(CFLAGS) -c build/cy_api.c -o build/cy_api.o
# 
build/main.o:
	# cp src/main.c build/main.c # Force adding main.c to build
	$(CC) $(CFLAGS) -c src/main.c -o build/main.o
	
build/circular_buffer.o:
	$(CC) $(CFLAGS) -c src/circular_buffer.c -o build/circular_buffer.o

build/udp.o:
	$(CC) $(CFLAGS) -c src/udp_receiver.c -o build/udp.o

build/antenna.o:
	$(CC) $(CFLAGS) -c src/antenna/antenna.c -o build/antenna.o

# 
# lib/beamformer.so: build/main.o build/cy_api.o
# 	$(CC) ${CFLAGS} -lm -shared -o lib/beamformer.so build/main.o build/cy_api.o -fPIC

# build/sampler.o: src/receiver.c
# 	$(CC) $(CFLAGS) -lm -c -o build/sampler.o src/receiver.c
# 
# lib/libsampler.so: build/sampler.o build/antenna.o
# 	$(CC) $(CFLAGS) -lm -shared -o lib/libsampler.so build/sampler.o build/antenna.o -fPIC

build/sampler.o: src/main.c
	$(CC) $(CFLAGS) -lm -c -o build/sampler.o src/main.c

build/config.o:
	python3 ./src/build_config.py

lib/libsampler.so: build/config.o build/sampler.o build/antenna.o build/circular_buffer.o build/udp.o 
	$(CC) $(CFLAGS) -lm -shared -o lib/libsampler.so build/sampler.o build/antenna.o build/udp.o build/circular_buffer.o -fPIC


run: build/main.o build/circular_buffer.o build/udp.o# build/cy_api.o 
	$(CC) build/main.o build/circular_buffer.o build/udp.o $(LFLAGS) -o $(BIN)
	# $(CC) build/main.o $(LFLAGS) -o $(BIN)
.PHONY: clean

# $(BUILD)/*.c $(BUILD)/*.h
clean:
	echo "Removing Generated Build Files"
	
	rm $(BUILD)/*.o 

	echo "Removing Binaries"
	rm $(BIN)

	echo "Removing Libraries"
	rm lib/*.so