/******************************************************************************
 * Title                 :   Perform a delay of a signal
 * Filename              :   antenna/delay.c
 * Author                :   Irreq
 * Origin Date           :   20/06/2023
 * Version               :   3.0.0
 * Compiler              :   gcc (GCC) 9.5.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 Functions to delay signals

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <immintrin.h>

#include "../config.h"

/*

gcc delay.c -ffinite-math-only -lm -march=native -mavx2 -O3 -o run && ./run

*/

#define OFFSET N_TAPS / 2

#define PI 3.14159265359

#define SSE_SIMD_LENGTH 4
#define AVX_SIMD_LENGTH 8

#define KERNEL_LENGTH 16
#define VECTOR_LENGTH 16
#define ALIGNMENT 32 // Must be divisible by 32

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
    // __m256 kernel_block __attribute__((aligned(ALIGNMENT)));

    __m256 aligned_kernel[N_TAPS] __attribute__((aligned(ALIGNMENT)));

    // Repeat the kernel across the vector
    for (int i = 0; i < N_TAPS; i++)
    {
        aligned_kernel[i] = _mm256_broadcast_ss(h + i);
    }

    
    //float *padded = (float *) _mm_malloc((N_SAMPLES + N_TAPS) * sizeof(float), ALIGNMENT);
    //memset(padded, 0, (N_SAMPLES + N_TAPS) * sizeof(float));

    // Allocate memory for the padded signal
    float data[N_SAMPLES + N_TAPS] __attribute__((aligned(ALIGNMENT))) = {0.0};
    float *padded __attribute__((aligned(ALIGNMENT))) = &data[0];

    //// Zero entire buffer
    //for (int i = 0; i < N_SAMPLES + N_TAPS; i++)
    //{
    //    padded[i] = 0.0;
    //}

    // Copy the original array to the padded array
    memcpy(padded + OFFSET, signal, N_SAMPLES * sizeof(float));

    __m256 accumulator0 __attribute__((aligned(ALIGNMENT)));
    __m256 accumulator1 __attribute__((aligned(ALIGNMENT)));

    for (int i = 0; i < N_SAMPLES; i += 2 * AVX_SIMD_LENGTH)
    {
        accumulator0 = _mm256_setzero_ps();
        accumulator1 = _mm256_setzero_ps();

        for (int k = 0; k < N_TAPS; k++)
        {
            data_block = _mm256_loadu_ps(padded + i + k);
            accumulator0 = _mm256_fmadd_ps(data_block, aligned_kernel[k], accumulator0);

            data_block = _mm256_loadu_ps(padded + i + k + AVX_SIMD_LENGTH);
            accumulator1 = _mm256_fmadd_ps(data_block, aligned_kernel[k], accumulator1);
        }

        //_mm256_storeu_ps(out + i, accumulator0);
        //_mm256_storeu_ps(out + i + AVX_SIMD_LENGTH, accumulator1);
        _mm256_storeu_ps(out + i, accumulator0);
        _mm256_storeu_ps(out + i + AVX_SIMD_LENGTH, accumulator1);
    }

    //_mm_free(padded);
}

// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
float sum8(__m256 x)
{
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}


void delay_vectorized_add(float *signal, float *h, float *out)
{
    // Allign data
    __m256 data_block __attribute__((aligned(ALIGNMENT)));
    __m256 kernel_block __attribute__((aligned(ALIGNMENT)));

    // float padded[N_SAMPLES + N_TAPS];
    float *padded = (float *)_mm_malloc((N_SAMPLES + N_TAPS) * sizeof(float), ALIGNMENT);
    // float *padded = (float *)aligned_alloc(ALIGNMENT, (N_SAMPLES + N_TAPS) * sizeof(float));
    // Initialize the padded array with zeros
    memset(padded, 0, (N_SAMPLES + N_TAPS) * sizeof(float));
    // Copy the original array to the padded array
    memcpy(padded + OFFSET, signal, N_SAMPLES * sizeof(float));

    float *kernel = (float *)_mm_malloc((N_TAPS) * sizeof(float), ALIGNMENT); //(float *)aligned_alloc(ALIGNMENT, (N_TAPS) * sizeof(float));
    memcpy(kernel, h, N_TAPS * sizeof(float));

    __m256 accumulator __attribute__((aligned(ALIGNMENT)));

    for (int i = 0; i < N_SAMPLES; i++)
    {
        accumulator = _mm256_setzero_ps();

        for (int k = 0; k < N_TAPS; k += AVX_SIMD_LENGTH)
        {
            data_block = _mm256_loadu_ps(padded + i + k);
            kernel_block = _mm256_load_ps(kernel + k);
            accumulator = _mm256_fmadd_ps(data_block, kernel_block, accumulator);
        }

        //out[i] = hsum256_ps_avx(accumulator);
        out[i] += sum8(accumulator);

        // _mm256_storeu_ps(out + i, acc);
    }

    // Free the allocated memory
    _mm_free(padded);
    _mm_free(kernel);
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

void validate(float *signal, float *h)
{
    float *out_vectorized = (float *)aligned_alloc(ALIGNMENT, N_SAMPLES * sizeof(float));
    delay_vectorized(signal, h, out_vectorized);

    float *out_naive = (float *)aligned_alloc(ALIGNMENT, N_SAMPLES * sizeof(float));
    delay_naive(signal, h, out_naive);

    for (size_t i = 0; i < N_SAMPLES; i++)
    {
        printf("%f %f\n", out_vectorized[i], out_naive[i]);
    }

    printf("\n");
}

void bench_mark(float *signal, float *h)
{

    float *out = (float *)aligned_alloc(ALIGNMENT, N_SAMPLES * sizeof(float));

    int test_samples = 10000;

    // Perform convolution using vectorized implementation
    clock_t start = clock();
    for (int i = 0; i < test_samples; i++)
    {
        delay_vectorized(signal, h, out);
    }
    clock_t end = clock();
    double vectorized_time = (double)(end - start) / CLOCKS_PER_SEC; // / test_samples;

    // Perform convolution using naive implementation
    start = clock();
    for (int i = 0; i < test_samples; i++)
    {
        delay_naive(signal, h, out);
    }
    end = clock();

    double naive_time = (double)(end - start) / CLOCKS_PER_SEC; // / test_samples;

    printf("Vectorized Convolution Execution Time: %f seconds\n", vectorized_time);
    printf("Naive Convolution Execution Time: %f seconds\n", naive_time);

    printf("Vetctorized code is %f times faster\n", naive_time/vectorized_time);
}

int main(int argc, char const *argv[])
{
    float *signal = malloc(N_SAMPLES * sizeof(float));
    for (int i = 0; i < N_SAMPLES; i++)
    {
        signal[i] = (float)i;
    }

    float *h = malloc(N_TAPS * sizeof(float));

    generate_coefficients(0.5, h);

    validate(signal, h);
    bench_mark(signal, h);

    return 0;
}