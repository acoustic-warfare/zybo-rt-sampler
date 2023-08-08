
#include <math.h>
#include <string.h> // malloc
#include <stdlib.h> // free
#include "config.h"
#include "lerp_and_sum.h"

#define LERP 0



int *whole_samples_lerp;
float *fractional_samples_lerp;


/**
 * @brief fast LERP delay 
 * 
 * WARNING you must convert the fraction to: h := 1 - h
 * 
 * @param signal 
 * @param out 
 * @param h 
 * @param pad 
 */
inline void lerp_delay(float *signal, float *out, float h, int pad)
{
    for (int i = 0; i < N_SAMPLES - pad - 1; i++)
    {
        out[pad + i + 1] += signal[i] + h * (signal[i + 1] - signal[i]); // Must precalc h = 1 - h
    }
}

void miso_lerp(float *signals, float *out, int *adaptive_array, int n, int offset)
{
    // Reset the output for the new direction
    memset(out, 0, (N_SAMPLES) * sizeof(float));
    
    int i, pos_mic;

    for (i = 0; i < n; i++)
    {
        pos_mic = adaptive_array[i];

        // Perform the delay of the input signal
        lerp_delay(
            signals + pos_mic * N_SAMPLES,               // Signal
            out,                                         // Out buffer
            fractional_samples_lerp[offset + i],  // interpolation factor (fractional delay)
            whole_samples_lerp[offset + i]       // Padding
        );
    }

    // // Get the average
    // for (i = 0; i < N_SAMPLES; i++)
    // {
    //     out[i] /= (float)n;
    // }
}

void mimo_lerp(float *signals, float *image, int *adaptive_array, int n)
{
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
            miso_lerp(signals, &out[0], adaptive_array, n, y_offset + x_offset);

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


void load_coefficients_lerp(float *delays, int n)
{
    int i;
    double pad;

    whole_samples_lerp = (int *)malloc(n * sizeof(int));
    fractional_samples_lerp = (float *)malloc(n * sizeof(float));

    for (i = 0; i < n; i++)
    {
        // Due to optimization we must reverse the fractional part
        fractional_samples_lerp[i] = 1.0 - (float)modf(delays[i], &pad);
        whole_samples_lerp[i] = (int)pad;
    }
    
}

void unload_coefficients_lerp()
{
    free(whole_samples_lerp);
    free(fractional_samples_lerp);
}

#if LERP

#include <stdio.h>

int main(int argc, char const *argv[])
{
    /* code */

    float signal[N_SAMPLES], out[N_SAMPLES];
    for (int i = 0; i < N_SAMPLES; i++)
    {
        signal[i] = (float)i / 3;
        out[i] = 0.0;
    }

    lerp_delay(&signal[0], &out[0], 1 - 0.00001, 0);

    printf("\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%f %f \n",signal[i], out[i]);
    }


    load_coefficients_lerp(&signal[0], N_SAMPLES);
    for (int i = 0; i < 20; i++)
    {
        printf("%d %f\n", whole_samples_lerp[i], fractional_samples_lerp[i]);
    }

    unload_coefficients_lerp();

    return 0;
}
#endif
