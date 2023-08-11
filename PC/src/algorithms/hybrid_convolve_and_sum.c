/******************************************************************************
 * Title                 :   A convolve- and sum beamformer
 * Filename              :   src/algorithms/convolve_and_sum.c
 * Author                :   Irreq
 * Origin Date           :   20/07/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 11.3.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 This file is a delay and sum beamformer which uses a convolution to delay
 the signals with a kernal similar to a sinc filter in order
 to achieve true-time delay. One caveat with this implementation is that
 in order to delay a signal N samples, one must calculate a kernel with atleast
 2N Coefficients, since the algorithm can only delay up to half of the number of
 coefficients (h).

 Worst case scenario:

 MAX_RES_X * MAX_RES_Y * n * N_SAMPLES * N_TAPS

 Which may result in a time-complexity of O(n^5) // Different `n`

*/

#include <string.h>
#include <math.h>

#include <immintrin.h>

#include "config.h"

#define OFFSET N_TAPS / 2

// #ifndef DEBUG
// #define DEBUG 1
// #endif

#define DEBUG 0

int *whole_samples_convolve;
float *convolve_coefficients_fractional;

// VECTORIZED

#define AVX_SIMD_LENGTH 8 // AVX2 m256 width
#define ALIGNMENT 32      // Must be divisible by 32


void convolve_hybrid_delay_add(float *signal, float *h, int pad, float *out)
{
    float padded[N_SAMPLES + N_TAPS] = {0.0};

    memcpy(&padded[OFFSET], signal, sizeof(float) * N_SAMPLES);

    for (int i = 0; i < N_SAMPLES - pad - 1; i++)
    {
        for (int k = 0; k < N_TAPS; k++)
        {
            out[pad + i + 1] += h[k] * padded[i + k];
        }
    }
}

void miso_convolve_hybrid(float *signals, float *out, int *adaptive_array, int n, int offset)
{
    // Reset the output for the new direction
    memset(out, 0, (N_SAMPLES) * sizeof(float));

    int i, pos_mic;

    for (i = 0; i < n; i++)
    {
        pos_mic = adaptive_array[i];

        // Perform the delay of the input signal
        convolve_hybrid_delay_add(
            signals + pos_mic * N_SAMPLES, // Signal

            &convolve_coefficients_fractional[(offset + i) * N_TAPS], // interpolation factor (fractional delay)
            whole_samples_convolve[offset + i],            // Padding
            out                                          // Out buffer
        );
    }
}

void mimo_convolve_hybrid(float *signals, float *image, int *adaptive_array, int n)
{
    // dummy output
    float out[N_SAMPLES];
    float sum;

    // Since the delay coefficients are 1D, we need to point to the correct
    // position in the delay vector.
    int x_offset, y_offset;

    for (int y = 0; y < MAX_RES_Y; y++)
    {
        y_offset = y * MAX_RES_X * n;
        for (int x = 0; x < MAX_RES_X; x++)
        {
            x_offset = x * n;

            // Get the signal from the current direction
            miso_convolve_hybrid(signals, &out[0], adaptive_array, n, y_offset + x_offset);

            // Compute the mean power of the signal
            sum = 0.0;
            for (int k = 0; k < N_SAMPLES; k++)
            {
                out[k] /= (float)n; // Divide by number of microphones
                sum += powf(out[k], 2);
            }

            sum /= (float)N_SAMPLES;

            image[y * MAX_RES_X + x] = sum;
        }
    }
}

#define PI 3.14159265359
void compute_h_convolve(float *h, double delay)
{
    double sum = 0.0;
    double epsilon = 1e-9;

    // This is the crucial math
    double tau = 0.5 - delay + epsilon;

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

        sum += h_i_d;

        h[i] = (float)h_i_d;
    }

    for (int i = 0; i < N_TAPS; i++)
    {
        // Normalize to get unity gain.
        h[i] /= (float)sum;
    }
}


#include <stdio.h>
void load_coefficients_convolve_hybrid(float *h, int n)
{
    convolve_coefficients_fractional = (float *)malloc(n * N_TAPS * sizeof(float));
    whole_samples_convolve = (int *)malloc(n * sizeof(int));

    int i;
    double pad, fraction;

    for (i = 0; i < n; i++)
    {
        // Due to optimization we must reverse the fractional part
        fraction = 1.0 - modf((double)h[i], &pad);
        whole_samples_convolve[i] = (int)pad;

        for (int k = 0; k < N_TAPS; k++)
        {
            compute_h_convolve(&convolve_coefficients_fractional[i * N_TAPS], fraction);
        }
    }
}

void unload_coefficients_convolve_hybrid()
{
    free(convolve_coefficients_fractional);
    free(convolve_coefficients_fractional);
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
//     convolve_coefficients_fractional = (float *)malloc(MAX_RES_X * MAX_RES_Y * ACTIVE_ARRAYS * ROWS * COLUMNS * N_TAPS * sizeof(float));

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

//             compute_antenna(azimuth, elevation, convolve_coefficients_fractional + y_offset + x_offset);
//         }

//     }

// }