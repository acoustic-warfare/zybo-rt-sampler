This is a collection of backend MIMO-, MISO- and delay algorithms which uses different techniques in order to achieve beamforming using delay and sum in the time domain.

Each algorithm consist of the following functions:

* **delay** Delays the signal for a specific number of samples (can be fractional)

* **miso** Multiple inputs single output. Performs the delay for a specific amount for each microphone in a specific direction.

* **mimo** Multiple inputs multiple outputs. Generates a heatmap of the computed power levels of signals in for all precalculated directions.


## convolve_and_sum.c [AVX256]
A beamformer that uses sinc filter for true time delay of the signals. This sinc filter is then convolved over a padded signal which introduces a delay in the timedomain. To speed up performance AVX256 instructions such as `FMA` is being used for to compute the convolution in parallel.

## hybrid_convolve_and_sum.c
A hybrid beamformer that uses both zero-padding and sinc-filter convolution to delay the signal.

## lerp_and_sum.c
A beamformer which uses both zero-padding and and naive linear interpolation for calculating the time delay.

## pad_and_sum.c
A beamformer that only uses zero-padding for delaying the signal. This is the simplest for of delaying the signal since no actual computation is requried for the delay step.


# Future improvements

1. Create a hybrid algorithm which utilizes the efficiency of the pad-beamformer with the precision of the convolve-beamformer for a final algorithm that is both fast and precise.

2. Implement in C, a beamformer that performs in the frequency domain.

3. Vectorize the code using SIMD instructions and or offload computation to the GPU 