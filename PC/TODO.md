Dear future programmer, you may wonder why on earth this program is structured this way, but we would like to separate the methods out of `src/api.c`. We were unable to compile the program with the different methods in different files, that is why a single library `lib/beamformer.so` is being created. We were unable to have Python (Cython) to keep the memory in a shared scope when we worked with the beamforming and audio playback.

# Known bugs
We have found some bugs that we *believe* stems from wrong shared-memory usage.
The program initiates several processes and we choosed shared-memory way of IPC. When the program ***crashes*** it is imperative that correct clean-up is made.
This is not always the case when dealing with the higher level python libraries that live on top of the C code. We use OpenCV for camera feed and image rendering, and PortAudio for audio playback. If any of those dependencies crash during runtime, it is sadly discarded as undefined behavior.

# Repeated Code
The time-domain beamforming algorithms are actually quite similar, the only thing that sets them apart is the delay function. However, at the moment, each file esposes both MISO and MIMO functions, even though they are "the same". It would be quite easy to create `#if` statements for which algorithm to use in a single MIMO/MISO function, this however won't allow the user to run different implementations at the same time.

# Optimization
We used enterprise grade Dell computers with an Intel Core i5, this allowed use to use AVX256 SIMD instructions for convolution, resulting in a 15x performance boost compared to regular gcc `-O3` optimization, however, the gain *should* be better if performing the beamforing on a GPU instead. The inline delay functions are kernel-ready.

# Python overhead
The single reason for using python in this application is so that a application window with callbacks can be used. It has been noticed that compiling using the Cython compiler, the Cython/Python code down to C gives the program a hefty performance hit. This is not an issue when dealing with offline data, but for realtime computation is quite expensive.

# Cython
Compilation using cython was unable to put multiple files as inline, therefore all cython code that requires the same variables are located in a single file. This results in repeated code and large files.

Since `cdef` are not allowed in the global scope or inside classes, we had to resort to creating wrapper functions that declared all variables and had the main loop located inside the function paired with a boolean shared value in order to stop the functions and free the allocated memory. 

# Low-level Future work:

## Padding -> read previous samples
At the moment, a naive delay is made by putting zeros infront of the signal in the amount that it needs to be delayed. However, this introduces clipping in the signal as the data might have to jump from it's initial value X to zero in an instant. This only works when the signal begins at zero, which it oftentimes do not. A better solution is to read the last N samples of the previous signal as the delay of the current signal. A working ring buffer is already implemented but has not been put in place for this work.

## UDP handshake
The receiver tunes in to the UDP datagram stream, but it has now way of controlling or see how the packages looks like, therefore in order to receive the correct message size, we had to define (hard-coded) on both the FPGA and the PC on how many arrays are to be transmitted. This is a wasteful since if we only need one array, 4 arrays are actually sent. This should instead be implemented that the PC transmits a single packet on what sample-rate, and number of arrays to receive. This will also reduce the total packet size, but that might be marginal.

## UDP playback
We initially used wireshark to replay the receive messages, but it would be beneficial to write a little program that can replay the packets in realtime. (Debug purposes)

## Peak detection
There was an attempt at using 2D convolution in order to find local maxima in the image for detecting multiple sources. This failed when the resolution went up due to there being multiple pixels in the viscinity being both higher and lower which gave the false impression having more peaks than there was on the grand scale.

## Implement the frequency-domain algorithm in pure C
The frequency-domain beamformer is currently implemented in Python using vectorized NumPy Matrix calculations however, the constant copying of data between C and python introduces overhead for the program. We are using contigious memory buffers in the Cython->Python API, however NumPy uses memory checks (good) but that is not necessary when we know the structure of the data down to the byte, so unecessary checks are done in Python.

# High-level Future Work

## Tracking + Prediction
The current implementation can barely detect a signal, that is, it can only detect power-levels in a specific direction not actually keep track of where it is located. A 3D Kalman filter has been created located in `src/kf.hpp` which is capable of both tracking the current sample but also able to predict future location of the signal albeit not so good after 5 time-steps in the future.

Maybe a ML approach to prediction can be used.

## Improved spatial filtering
The delay-and-sum beamformers use only constructive interference for generating a beam (and destructive as well). This is not an issue with single sources and correct angles an no reflections, but if we were to steer against a side-lobe or greating-lobe, that would also create a detectable peak. One could use techniques for supressing signals from undesired directions



