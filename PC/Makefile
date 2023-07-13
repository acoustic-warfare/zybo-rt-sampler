CC = gcc
CYCC = cython3

CFLAGS = `python3-config --cflags`
CFLAGS += -fPIE

# For x86_64 SIMD
CFLAGS += -Ofast -flto -ffast-math -march=native -mno-vzeroupper -msse3 -msse4 -mavx2 -mavx
# get only 3.XX
PYTHON_VERSION = $(python3 --version | grep -Po "\d+\.\d+" | echo)
PYTHON_MODULES = -I/usr/lib/python$(PYTHON_VERSION)/site-packages/numpy/core/include/

CFLAGS += -Ibuild/ -Isrc/ # Add build to path

CFLAGS += $(PYTHON_MODULES)
LFLAGS = `python3-config --embed --ldflags` # Must be --embed for >python3.8

LFLAGS += $(PYTHON_MODULES)
BIN = bin
EXECUTABLE = run

BUILD = build
PYTHON_PATH = `python3 -c "import distutils.sysconfig;print(distutils.sysconfig.get_python_inc())"`

.PHONY: all

all: cybeamformer lib/beamformer.so lib/libsampler.so #run lib/libmiso.so lib/beamformer.so


config:
	python3 ./src/build_config.py

build/config.o:
	python3 ./src/build_config.py

cybeamformer: build/config.o
	cp -r src/* build/
	
	$(CC) -O3 --shared -fPIC -march=native -mavx2 -lm `ls build/*.c | grep -xv "build/main.c"` build/antenna/*.c -o build/delay.so
	
	$(CYCC) src/wrapper.pyx -o build/wrapper.c
	$(CC) -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -lm -march=native -mavx2 -I $(PYTHON_PATH) -o lib/wrapper.so build/wrapper.c build/delay.so
	#python3 -c "from lib.wrapper import *;main()"

## Cython stuff
#build/cy_api.c:
#	cp src/data/*.c build/
#	$(CYCC) -o build/cy_api.c src/data/cy_api.pyx
#
#build/cy_api.o: build/cy_api.c
#	#cp src/data/*.c build/
#	$(CC) $(CFLAGS) -c build/cy_api.c -o build/cy_api.o
#
#build/cy_main.o:
#	#cp src/data/cy_main.c build/cy_main.c
#	$(CC) $(CFLAGS) -c src/data/cy_main.c -o build/cy_main.o

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

build/delay.o:
	$(CC) $(CFLAGS) -lm -c -o build/delay.o src/antenna/delay.c

build/miso.o:
	$(CC) $(CFLAGS) -lm -c -o build/miso.o src/antenna/miso.c

build/antenna.o:
	$(CC) $(CFLAGS) -lm -c -o build/antenna.o src/antenna/antenna.c

build/beamformer.o:
	$(CC) $(CFLAGS) -lm -c -o build/beamformer.o src/beamformer.c

lib/libsampler.so: build/config.o build/sampler.o build/antenna.o build/circular_buffer.o build/udp.o 
	$(CC) $(CFLAGS) -lm -shared -o lib/libsampler.so build/sampler.o build/antenna.o build/udp.o build/circular_buffer.o -fPIC


lib/libmiso.so: build/config.o build/miso.o build/antenna.o build/circular_buffer.o build/udp.o build/delay.o
	$(CC) $(CFLAGS) -lm -shared -o lib/libmiso.so build/miso.o build/antenna.o build/circular_buffer.o build/udp.o build/delay.o -fPIC

lib/beamformer.so: build/config.o build/beamformer.o build/antenna.o build/delay.o build/circular_buffer.o build/udp.o 
	$(CC) $(CFLAGS) -O3 -march=native -mavx2 -lm -shared -o lib/beamformer.so build/beamformer.o build/antenna.o build/udp.o build/circular_buffer.o build/delay.o -fPIC

#run: build/main.o build/circular_buffer.o build/udp.o# build/cy_api.o 
#	$(CC) build/main.o build/circular_buffer.o build/udp.o $(LFLAGS) -o $(BIN)
#	# $(CC) build/main.o $(LFLAGS) -o bin/$(BIN)

#run: build/cy_api.o build/cy_main.o
#	$(CC) build/cy_main.o build/cy_api.o $(LFLAGS) -o $(BIN)/$(EXECUTABLE)


cython_project:
	python3 setup.py build_ext --inplace --build-lib=build
	mv *.so lib/
	touch lib/__init__.py

	echo "#!/usr/bin/env python3" >> run
	echo "# -*- coding: utf-8 -*-" >> run
	echo "# DO NOT EDIT. THIS IS CREATED DURING COMPILATION"
	echo "from lib import cy_api" >> run
	echo "cy_api.entrypoint()" >> run
	chmod +x run

	cp interface/config.py config.py

.PHONY: clean

clean:
	echo "Removing Generated Build Files"
	ls build/ | grep -xv ".gitignore" | xargs -I {} sh -c "rm -r build/{}"

# $(BUILD)/*.c $(BUILD)/*.h
clean_old:
	echo "Removing Generated Build Files"

	#ls $(BUILD)/ | grep -xv ".gitignore" | xargs rm
	#ls lib/ | grep -xv ".gitignore" | xargs rm
	#ls bin/ | grep -xv ".gitignore" | xargs rm

	ls $(BUILD)/ | grep -xv ".gitignore" | xargs -I {} sh -c "rm -r $(BUILD)/{}"
	
	rm -f run || true
	#rm $(BUILD)/*.o
	#rm $(BUILD)/*.h 
	#rm $(BUILD)/*.c
#
	#echo "Removing Libraries"
	##rm lib/*.so
#
	#echo "Removing Binaries"
	#rm bin/$(EXECUTABLE)