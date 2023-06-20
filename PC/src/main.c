#include <stdio.h>
#include <stdlib.h>

#include "udp_receiver.h"
#include "config.h"
#include "circular_buffer.h"



void test_rb()
{
    // Initialize the ring buffer
    ring_buffer *myRingBuffer = (ring_buffer *)calloc(1, sizeof(ring_buffer));
    myRingBuffer->index = 0;

    // float d[10] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    float d[BUFFER_LENGTH * 2];

    for (int i = 0; i < BUFFER_LENGTH * 2; i++)
    {
        d[i] = (float)(i + 1);
    }

    float *ptr = d;

    write_buffer(myRingBuffer, ptr, BUFFER_LENGTH, 0);

    write_buffer(myRingBuffer, ptr, BUFFER_LENGTH - 4, 0);

    float out[BUFFER_LENGTH] = {0.0};

    for (size_t i = 0; i < BUFFER_LENGTH; i++)
    {
        out[i] = 0.0;
    }

    // convolve_avx_unrolled_vector_unaligned_fma()

    read_buffer_mcpy(myRingBuffer, &out[0]);

    for (int i = 0; i < BUFFER_LENGTH; i++)
    {
        printf("%f ", out[i]);
    }

    printf("\n");
}

int main(int argc, char const *argv[])
{
    printf("Hello World! Number of samples: %d\n", N_SAMPLES);

    test_rb();
    return 0;
}
