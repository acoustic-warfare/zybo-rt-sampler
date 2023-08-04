/******************************************************************************
 * Title                 :   A pad- and sum beamformer
 * Filename              :   src/algorithms/pad_and_sum.c
 * Author                :   Irreq
 * Origin Date           :   20/07/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 11.3.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 This file is a delay and sum beamformer which pads the signals with zeros as
 the delay operation. This is fast as only a single summation is required.

 Worst case scenario:

 MAX_RES_X * MAX_RES_Y * n * N_SAMPLES

 Which may result in a time complexity of O(n^4) // Different `n`

*/

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "config.h"

int *whole_samples_h;  // The 1D delay coefficients
int *whole_miso_samples;

// #ifndef DEBUG
// #define DEBUG 1
// #endif

#define DEBUG 0

/*
Perform a delay
*/
void pad_delay(float *signal, float *out, int pos_pad)
{
    for (int i = 0; i < N_SAMPLES - pos_pad; i++)
    {
        out[pos_pad + i] += signal[i];
    }
}

/*
Perform a MISO for a certain direction

int offset: if you need coefficients in a specific direction = (y * MAX_RES_X * n + x * n)
*/
void miso_pad(float *signals, float *out, int *adaptive_array, int n, int offset)
{
    // Reset the output for the new direction
    memset(out, 0, (N_SAMPLES) * sizeof(float));

    int pos_pad, pos_mic;

    for (int m = 0; m < n; m++)
    {
        pos_mic = adaptive_array[m]; // Which mic to use
        pos_pad = whole_samples_h[offset + m]; // Delay amount
        // pos_pad = whole_samples_h[offset + pos_mic]; // Delay amount
        pad_delay(signals + pos_mic * N_SAMPLES, out, pos_pad);
        // pad_delay(&signals[pos_mic * N_SAMPLES], out, pos_pad);
    }
    
}

/*
Perform a MISO for a certain direction

int offset: if you need coefficients in a specific direction = (y * MAX_RES_X * n + x * n)
*/
void miso_pad2(float *signals, float *out, int *adaptive_array, int n, int offset)
{
    // Reset the output for the new direction
    memset(out, 0, (N_SAMPLES) * sizeof(float));

    int pos_pad, pos_mic;

    for (int m = 0; m < n; m++)
    {
        pos_mic = adaptive_array[m];           // Which mic to use
        pos_pad = whole_miso_samples[pos_mic]; // Delay amount

        pad_delay(signals + pos_mic * N_SAMPLES, out, pos_pad);
        // pad_delay(&signals[pos_mic * N_SAMPLES], out, pos_pad);
    }
}

/*
Perform a MIMO

As this algorithm uses a 1D flattened array for mic delays, 
a offset is calculated based on which microphone and direction is currently being calculated
*/
void mimo_pad(float *signals, float *image, int *adaptive_array, int n)
{

#if DEBUG
    int progress = 0;
    printf("Status mimo pad\n");
#endif

    // dummy output
    float out[N_SAMPLES];
    float sum;

    int x_offset, y_offset;

    for (int y = 0; y < MAX_RES_Y; y++)
    {
        y_offset = y * MAX_RES_X * n;
        for (int x = 0; x < MAX_RES_X; x++)
        {
            x_offset = x * n;
            miso_pad(signals, &out[0], adaptive_array, n, y_offset + x_offset);

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
            printf("\r%f\%     ", (float)progress / (float)(MAX_RES_X * MAX_RES_Y) * 100);
#endif
        }

#if DEBUG
        printf("\n");
#endif
    }
}


#include <stdio.h>
void load_coefficients_pad(int *whole_samples, int n)
{
    whole_samples_h = (int*) malloc(n * sizeof(int));
    memcpy(whole_samples_h, whole_samples, n * sizeof(int));
}

void load_coefficients_pad2(int *whole_miso, int n)
{
    whole_miso_samples = (int*) malloc(n * sizeof(int));
    memcpy(whole_miso_samples, whole_miso, n * sizeof(int));
}

void unload_coefficients_pad()
{
    free(whole_samples_h);
}

void unload_coefficients_pad2()
{
    free(whole_miso_samples);
}

// /*

// MISO using array padding

// Complexity: O(n^2)

// float *signal: your signal data
// int *adaptive_array: array consisting of which mic index to use
// float *out: output array, same characteristics as signal
// int n: size of adaptive_array
// int offset: if you need coefficients in a specific direction = (y * MAX_RES_X * n + x * n)
// */
// void miso2_delay(float *signal, int *adaptive_array, float *out, int n, int offset)
// {
//     // Reset the output for the new direction
//     memset(out, 0, (N_SAMPLES) * sizeof(float));

//     int pos, pos_u;

//     for (int s = 0; s < n; s++)
//     {
//         pos_u = adaptive_array[s];

//         pos = whole_samples_h[offset + s];
//         for (int i = 0; i < N_SAMPLES - pos; i++)
//         {
//             out[pos + i] += signals[pos_u * N_SAMPLES + i];
//         }
//     }
// }




