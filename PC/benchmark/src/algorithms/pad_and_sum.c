#include <stdlib.h>

#include "config.h"

int *whole_samples_h;

// #ifndef DEBUG
// #define DEBUG 1
// #endif

#define DEBUG 1

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

        pad_delay(signals + pos_mic * N_SAMPLES, out, pos_pad);
    }
    
}

/*
Perform a MIMO
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

void unload_coefficients_pad()
{
    free(whole_samples_h);
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