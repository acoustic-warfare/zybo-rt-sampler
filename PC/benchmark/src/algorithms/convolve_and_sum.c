#include <string.h>

#include <immintrin.h>

#include "config.h"

#define OFFSET N_TAPS / 2

// #ifndef DEBUG
// #define DEBUG 1
// #endif

#define DEBUG 1

float *convolve_coefficients;

// VECTORIZED

#define AVX_SIMD_LENGTH 8 // AVX2 m256 width
#define ALIGNMENT 32      // Must be divisible by 32

/*
Naive convolution for adding to a single out with reset
*/
void _convolve_delay_naive(float *signal, float *h, float *out)
{
    float padded[N_SAMPLES + N_TAPS] = {0.0};

    memcpy(&padded[OFFSET], signal, sizeof(float) * N_SAMPLES);

    // 1D backwards convolution
    for (int i = 0; i < N_SAMPLES; i++)
    {
        out[i] = 0.0;

        for (int k = 0; k < N_TAPS; k++)
        {
            out[i] += h[k] * padded[i + k];
        }
    }
}

/*
Naive convolution for adding to a single out without reset
*/
void convolve_delay_naive_add(float *signal, float *h, float *out)
{
    float padded[N_SAMPLES + N_TAPS] = {0.0};

    memcpy(&padded[OFFSET], signal, sizeof(float) * N_SAMPLES);

    // 1D backwards convolution
    for (int i = 0; i < N_SAMPLES; i++)
    {
        for (int k = 0; k < N_TAPS; k++)
        {
            out[i] += h[k] * padded[i + k];
        }
    }
}

/*
Vectorized convolution for AVX2 with dual accumulators
*/
void convolve_delay_vectorized(float *signal, float *h, float *out)
{
    __m256 data_block __attribute__((aligned(ALIGNMENT)));
    __m256 aligned_kernel[N_TAPS] __attribute__((aligned(ALIGNMENT)));

    // Repeat the kernel across the vector
    for (int i = 0; i < N_TAPS; i++)
    {
        aligned_kernel[i] = _mm256_broadcast_ss(h + i);
    }

    float data[N_SAMPLES + N_TAPS] __attribute__((aligned(ALIGNMENT))) = {0.0};
    float *padded __attribute__((aligned(ALIGNMENT))) = &data[0];

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

        _mm256_storeu_ps(out + i, accumulator0);
        _mm256_storeu_ps(out + i + AVX_SIMD_LENGTH, accumulator1);
    }
}

// Borrowed code
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

/*
Vectorized convolution for AVX2 with single accumulator without reset
*/
void convolve_delay_vectorized_add(float *signal, float *h, float *out)
{
    __m256 data_block __attribute__((aligned(ALIGNMENT)));
    __m256 kernel_block __attribute__((aligned(ALIGNMENT)));

    float *padded = (float *)_mm_malloc((N_SAMPLES + N_TAPS) * sizeof(float), ALIGNMENT);

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

        out[i] += sum8(accumulator);
    }

    // Free the allocated memory
    _mm_free(padded);
    _mm_free(kernel);
}


// NAIVE

void convolve_delay_naive(float *signal, float *out, float *h)
{
    float padded[N_SAMPLES + N_TAPS] = {0.0};

    memcpy(&padded[OFFSET], signal, sizeof(float) * N_SAMPLES);

    // 1D backwards convolution
    for (int i = 0; i < N_SAMPLES; i++)
    {
        for (int k = 0; k < N_TAPS; k++)
        {
            out[i] += h[k] * padded[i + k];
        }
    }
}

void miso_convolve_naive(float *signals, float *out, int *adaptive_array, int n, int offset)
{
    // Reset the output for the new direction
    memset(out, 0, (N_SAMPLES) * sizeof(float));

    int pos_pad, pos_mic;

    float *h;

    for (int m = 0; m < n; m++)
    {
        pos_mic = adaptive_array[m];           // Which mic to use
        // h = &convolve_coefficients[offset + m * N_TAPS]; // Delay amount

        convolve_delay_naive(signals + pos_mic * N_SAMPLES, out, convolve_coefficients + offset + m * N_TAPS);
    }
}

void mimo_convolve_naive(float *signals, float *image, int *adaptive_array, int n)
{
    // dummy output
    float out[N_SAMPLES];
    float sum;

    int x_offset, y_offset;

    #if DEBUG
    int progress = 0;
    printf("Status mimo convolve naive\n");
    #endif

    for (int y = 0; y < MAX_RES_Y; y++)
    {
        y_offset = y * MAX_RES_X * n * N_TAPS;
        for (int x = 0; x < MAX_RES_X; x++)
        {
            x_offset = x * n * N_TAPS;
            miso_convolve_naive(signals, &out[0], adaptive_array, n, y_offset + x_offset);

            sum = 0.0;
            for (int k = 0; k < N_SAMPLES; k++)
            {
                out[k] /= (float)n; // Divide by number of microphones
                sum += powf(out[k], 2);
            }

            sum /= (float)N_SAMPLES;

            image[y * MAX_RES_X + x] = sum;
#if DEBUG
            progress++;
            printf("\r%f\%     ", (float)progress/(float)(MAX_RES_X * MAX_RES_Y)*100);
#endif
        }

#if DEBUG
        printf("\n");
#endif
    }
}



void miso_convolve_vectorized(float *signals, float *out, int *adaptive_array, int n, int offset)
{
    // Reset the output for the new direction
    memset(out, 0, (N_SAMPLES) * sizeof(float));

    int pos_pad, pos_mic;

    float *h;

    for (int m = 0; m < n; m++)
    {
        pos_mic = adaptive_array[m];            // Which mic to use
        h = &convolve_coefficients[offset + m]; // Delay amount

        convolve_delay_vectorized(signals + pos_mic * N_SAMPLES, out, h);
    }
}


void mimo_convolve_vectorized(float *signals, float *image, int *adaptive_array, int n)
{
    // dummy output
    float out[N_SAMPLES];
    float sum;

    int x_offset, y_offset;

    for (int y = 0; y < MAX_RES_Y; y++)
    {
        y_offset = y * MAX_RES_X * n * N_TAPS;
        for (int x = 0; x < MAX_RES_X; x++)
        {
            x_offset = x * n * N_TAPS;

            miso_convolve_vectorized(signals, &out[0], adaptive_array, n, y_offset + x_offset);

            sum = 0.0;
            for (int k = 0; k < N_SAMPLES; k++)
            {
                out[k] /= (float)n; // Divide by number of microphones
                sum += powf(out[k], 2);
            }

            sum /= (float)N_SAMPLES;

            image[y * MAX_RES_Y + x] = sum;
        }
    }
}

#include <stdio.h>
void load_coefficients_convolve(float *h, int n)
{
    convolve_coefficients = (float*) malloc(n * sizeof(float));
    memcpy(convolve_coefficients, h, n * sizeof(float));
}

void unload_coefficients_convolve()
{
    free(convolve_coefficients);
}

// void compute_antenna(float azimuth, float elevation, float *h)
// {
//     float smallest = 0.0;

//     int i = 0;

//     float delays[ROWS * COLUMNS]

//     for (int row = 0; row < ROWS; row++)
//     {
//         for (int col = 0; col < COLUMNS; col++)
//         {
//             /* code */
//         }
//     }
// }


// void calculate_coefficients(int steps)
// {
//     convolve_coefficients = (float *)malloc(MAX_RES_X * MAX_RES_Y * ACTIVE_ARRAYS * ROWS * COLUMNS * N_TAPS * sizeof(float));

//     int x_offset, y_offset;
//     float azimuth, elevation;

//     for (int y = 0; y < MAX_RES_Y; y++)
//     {
//         y_offset = y * MAX_RES_X * ACTIVE_ARRAYS * ROWS * COLUMNS * N_TAPS;
//         elevation = (float)MAX_ANGLE * 2 * (float)y / float(MAX_RES_Y) - (float)MAX_ANGLE;
//         for (int x = 0; x < MAX_RES_X; x++)
//         {
//             x_offset = x * ACTIVE_ARRAYS * ROWS * COLUMNS * N_TAPS;
//             azimuth = (float)MAX_ANGLE * 2 * (float)x / float(MAX_RES_X) - (float)MAX_ANGLE;

//             compute_antenna(azimuth, elevation, convolve_coefficients + y_offset + x_offset);
//         }
        
//     }
    
// }