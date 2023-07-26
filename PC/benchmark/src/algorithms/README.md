This is a collection of backend MIMO-, MISO- and delay algorithms which uses different techniques in order to achieve beamforming using delay and sum in the time domain.

# Future improvements

1. Create a hybrid algorithm which utilizes the efficiency of the pad-beamformer with the precision of the convolve-beamformer for a final algorithm that is both fast and precise.

2. Implement in C, a beamformer that performs in the frequency domain.

3. Vectorize the code using SIMD instructions and or offload computation to the GPU 