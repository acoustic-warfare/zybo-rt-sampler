#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>

#include <immintrin.h>

#define N_MICROPHONES 1
#define N_SAMPLES 32

#define BUFFER_LENGTH N_MICROPHONES * N_SAMPLES

#define ALIGNMENT 32

#define AVX_SIMD_LENGTH 8
#define VECTOR_LENGTH 16
#define KERNEL_LENGTH 16
#define SSE_SIMD_LENGTH 4

typedef struct _ring_buffer
{
    int index;
    float data[BUFFER_LENGTH];
} ring_buffer;

/**
Write data from an address `in` to a ring buffer you can specify offset
but most of the times, it will probably just be 0
*/
void write_buffer(ring_buffer *rb, float *in, int length, int offset)
{
    // -1 for our binary modulo in a moment
    int buffer_length = BUFFER_LENGTH - 1;
    int previous_item = rb->index;

    int idx;
    for (int i = 0; i < length; ++i)
    {
        // modulo will automagically wrap around our index
        idx = (i + previous_item) & buffer_length;
        rb->data[idx] = in[i + offset];
    }

    // Update the current index of our ring buffer.
    rb->index += length;
    rb->index &= BUFFER_LENGTH - 1;
}

/*
Retrieve the data in the ring buffer. You can specify the offset.
the length is the number of previous samples in descending order that will be put into out
on the offset location and onwards.
This may result in stack smashing if offset yields in out of bounds array
*/
void read_buffer(ring_buffer *rb, float *out, int length, int offset)
{
    int index = 0;
    for (int i = rb->index; i < BUFFER_LENGTH; i++)
    {
        if (BUFFER_LENGTH - length <= index)
        {
            out[index + offset - (BUFFER_LENGTH - length)] = rb->data[i];
        }

        index++;
    }

    for (int i = 0; i < rb->index; i++)
    {
        if (BUFFER_LENGTH - length <= index)
        {
            out[index + offset - (BUFFER_LENGTH - length)] = rb->data[i];
        }

        index++;
    }
}

void load_vector(ring_buffer *rb, __m256 *vec, int offset)
{
    // vec = _mm256_loadu_ps();
}

/*
Almost 100 times faster than the code above
*/
void read_buffer_mcpy(ring_buffer *rb, float *out)
{
    int first_partition = BUFFER_LENGTH - rb->index;

    float *data_ptr = &rb->data[0];

    memcpy(out, (void *)(data_ptr + rb->index), sizeof(float) * first_partition);
    memcpy(out + first_partition, (void *)(data_ptr), sizeof(float) * rb->index);
}

void test()
{
    // float out[BUFFER_LENGTH] = {0.0};
    //
    // float startTime = (float)clock() / CLOCKS_PER_SEC;
    //
    // for (int i = 0; i < 10000; i++)
    //{
    //    //write_buffer(myRingBuffer, ptr+(i%BUFFER_LENGTH), 1, 0);
    //    read_buffer_mcpy(myRingBuffer, &out[0]);
    //    //read_buffer(myRingBuffer, &out[0], BUFFER_LENGTH, 0);
    //}
    //
    //
    ///* Do work */
    //
    // float endTime = (float)clock() / CLOCKS_PER_SEC;
    //
    // float timeElapsed = endTime - startTime;
    //
    // printf("Time taken: %f\n", timeElapsed);
}



int main(int argc, char const *argv[])
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

    float out[10] = {0.0};

    for (size_t i = 0; i < 10; i++)
    {
        out[i] = 0.0;
    }

    //convolve_avx_unrolled_vector_unaligned_fma()
    

    read_buffer_mcpy(myRingBuffer, &out[0]);

    for (int i = 0; i < 10; i++)
    {
        printf("%f ", out[i]);
    }

    printf("\n");

    return 0;
}
