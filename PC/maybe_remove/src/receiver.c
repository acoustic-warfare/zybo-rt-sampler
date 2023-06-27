#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>
#include <math.h>

#include <immintrin.h>

#define N_MICROPHONES 1
#define N_SAMPLES 256

#define N_TAPS 128 // 256 // 16
#define OFFSET N_TAPS / 2

#define PI 3.14159265359

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


void demo()
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

#define SSE_SIMD_LENGTH 4
#define AVX_SIMD_LENGTH 8

#define KERNEL_LENGTH 16
#define VECTOR_LENGTH 16
#define ALIGNMENT 32 // Must be divisible by 32

float hsum_ps_sse3(__m128 v)
{
    __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

float hsum256_ps_avx(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow = _mm_add_ps(vlow, vhigh);             // add the low 128
    return hsum_ps_sse3(vlow);                  // and inline the sse3 version, which is optimal for AVX
    // (no wasted instructions, and all of them are the 4B minimum)
}

void generate_coefficients(float fractional, float *h)
{
    double epsilon = 1e-9; // Small number to avoid dividing by 0

    double tau = 0.5 + epsilon - (double)fractional; // The offsetted delay

    double sum = 0.0;

    for (int i = 0; i < N_TAPS; i++)
    {
        // Fractional delay with support to delay entire frames up to OFFSET
        double h_i_d = (double)i - ((double)N_TAPS - 1.0) / 2.0 - tau;

        // Compute the sinc value: sin(xπ)/xπ
        h_i_d = sin(h_i_d * PI) / (h_i_d * PI);

        // To get np.arange(1-M, M, 2)
        double n = (double)(i * 2 - N_TAPS + 1);

        // Multiply sinc value by Blackman-window (https://numpy.org/doc/stable/reference/generated/numpy.blackman.html)
        double black_manning = 0.42 + 0.5 * cos(PI * n / ((double)(N_TAPS - 1)) + epsilon) + 0.08 * cos(2.0 * PI * n / ((double)(N_TAPS - 1) + epsilon));

        h_i_d *= black_manning;

        h[i] = (float)h_i_d;

        sum += h_i_d;
    }

    // printf("%f\n", sum);

    float gain = (float)fabs(sum);

    // printf("%f\n", gain);

    for (int i = 0; i < N_TAPS; i++)
    {
        // Normalize to get unity gain.
        h[i] /= gain;
    }
}

/*
Delay a signal by creating a reversed convolution of size (N + M)
and return only of size N
*/
void delay_naive(float *signal, float *h, float *out)
{
    float *padded = malloc((N_SAMPLES + N_TAPS) * sizeof(float));

    // Zero entire buffer
    for (int i = 0; i < N_SAMPLES + N_TAPS; i++)
    {
        padded[i] = 0.0;
    }

    // Load offset signal to buffer
    for (int i = 0; i < N_SAMPLES; i++)
    {
        padded[i + OFFSET] = signal[i];
    }

    // 1D backwards convolution
    for (int i = 0; i < N_SAMPLES; i++)
    {
        out[i] = 0.0;

        for (int k = 0; k < N_TAPS; k++)
        {
            out[i] += h[k] * padded[i + k];
        }
    }

    free(padded);
}

void delay_vectorized(float *signal, float *h, float *out)
{
    __m256 data_block __attribute__((aligned(ALIGNMENT)));
    __m256 kernel_block __attribute__((aligned(ALIGNMENT)));
    __m256 accumulator __attribute__((aligned(ALIGNMENT)));

    //__m256 kernel[N_TAPS] __attribute__((aligned(ALIGNMENT)));

    //for (int i = 0; i < N_TAPS; i++)
    //{
    //    kernel[i] = _mm256_broadcast_ss(&h[i]);
    //}
//
    //data_block = _mm256_loadu_ps(signal);
    //kernel_block = _mm256_loadu_ps(h);
//
    //accumulator = _mm256_setzero_ps();
//
    //accumulator = _mm256_fmadd_ps(data_block, kernel_block, accumulator);
//
    //float res = 0.0;
    //for (int i = 0; i < 8; i++)
    //{
    //    res += accumulator[i];
    //    //printf("%f ", accumulator[i]);
    //    printf("%f ", kernel_block[i]);
    //}
//
    //float result = hsum256_ps_avx(accumulator);

    //float *padded[N_SAMPLES + N_TAPS];
    //// Initialize the padded array with zeros
    //memset(padded, 0, (N_SAMPLES + N_TAPS) * sizeof(float));
    //// Copy the original array to the padded array
    //memcpy(padded + OFFSET, signal, N_SAMPLES * sizeof(float));

    float *padded = malloc((N_SAMPLES + N_TAPS) * sizeof(float));

    //float *padded = (float *)aligned_alloc(ALIGNMENT, (N_SAMPLES + N_TAPS) * sizeof(float));

    memset(padded, 0, (N_SAMPLES + N_TAPS) * sizeof(float));
    // Copy the original array to the padded array
    memcpy(padded + OFFSET, signal, N_SAMPLES * sizeof(float));

    for (int i = 0; i < N_SAMPLES; i++)
    {
        accumulator = _mm256_setzero_ps();

        // Perform Fused-Multiply-Add (FMA) on each vector jump
        for (int k = 0; k < N_TAPS / AVX_SIMD_LENGTH; k++)
        {
            data_block = _mm256_loadu_ps(padded + i + AVX_SIMD_LENGTH * k);
            kernel_block = _mm256_loadu_ps(h + AVX_SIMD_LENGTH * k);
            accumulator = _mm256_fmadd_ps(data_block, kernel_block, accumulator);
        }

        // Store the height sum (vector sum) in out
        out[i] = hsum256_ps_avx(accumulator);
    }

    free(padded);
}



int main(int argc, char const *argv[])
{

    float *h = malloc(N_TAPS * sizeof(float));

    float *signal = malloc(N_SAMPLES * sizeof(float));

    generate_coefficients(0.5, h);

    for (int i = 0; i < N_SAMPLES; i++)
    {
        signal[i] = (float)i; // / (float)N_SAMPLES;
    }

    int test_samples = 10000;

    float *out = (float *)aligned_alloc(ALIGNMENT, N_SAMPLES * sizeof(float));

    // Perform convolution using vectorized implementation
    clock_t start = clock();
    for (int i = 0; i < test_samples; i++)
    {
        delay_vectorized(signal, h, out);
        // vectorizedConvolution(signals, kernels, result_vectorized, num_signals, signal_length, kernel_length);
    }
    clock_t end = clock();
    double vectorized_time = (double)(end - start) / CLOCKS_PER_SEC;// / test_samples;

    // Perform convolution using naive implementation
    start = clock();
    for (int i = 0; i < test_samples; i++)
    {
        delay_naive(signal, h, out);
        // delay_vectorized(signal, h, out);
        // naiveConvolution(signals, kernels, result_naive, num_signals, signal_length, kernel_length);
    }
    end = clock();

    double naive_time = (double)(end - start) / CLOCKS_PER_SEC;// / test_samples;

    printf("Vectorized Convolution Execution Time: %f seconds\n", vectorized_time);
    printf("Naive Convolution Execution Time: %f seconds\n", naive_time);

    //delay_vectorized(signal, h, out);

    float *out2 = (float *)aligned_alloc(ALIGNMENT, N_SAMPLES * sizeof(float));

    delay_naive(signal, h, out2);

    delay_vectorized(signal, h, out);

    for (int i = 0; i < N_SAMPLES; i++)
    {
        //printf("%f %f %f\n", signal[i], out[i], out2[i]);
    }

    free(h);
    free(signal);

    return 0;
}
