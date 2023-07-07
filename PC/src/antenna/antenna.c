/******************************************************************************
 * Title                 :   Calculate antenna coefficients for delay
 * Filename              :   antenna/antenna.c
 * Author                :   Irreq
 * Origin Date           :   20/06/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 9.5.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 Functions for python to call when calculating coefficients

*/

#include <math.h>
#include <stdlib.h>

#include "../config.h"

#define PI 3.14159265359

/*
Calculate the different coefficients depeding on the antenna's direction and size
for a single antenna
*/
void directional_antenna_delay_coefficients(double azimuth,   // Horizontal
                                            double elevation, // Vertical
                                            int columns,
                                            int rows,
                                            float distance, // Distance between elements
                                            float fs,
                                            float propagation_speed,
                                            float **coefficients)

{
    // Convert listen direction to radians
    double theta = azimuth * -(double)PI / 180.0;
    double phi = elevation * -(double)PI / 180.0;

    float x_factor = (float)(sin(theta));
    float y_factor = (float)(sin(phi));

    // Allocate antenna array
    float *antenna_array = malloc((columns * rows) * sizeof(float));

    int element_index = 0;

    float smallest = 0.0;

    // Create antenna in space with middle in origo (0, 0)
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < columns; col++)
        {
            float half = distance / 2.0;

            // Assign middle of array to origo
            float tmp_col = (float)col * distance - (float)columns * half + half;
            float tmp_row = (float)row * distance - (float)rows * half + half;

            float tmp_delay = tmp_col * x_factor + tmp_row * y_factor;

            // Update so there is always one element furthest from world at 0 i.e all other delays are greater
            if (tmp_delay < smallest)
            {
                smallest = tmp_delay;
            }

            antenna_array[element_index] = tmp_delay;

            element_index += 1;
        }
    }

    // Create a delay map
    for (int i = 0; i < rows * columns; i++)
    {
        // Make the furthest element from source direction have no delay
        if (smallest < 0.0)
        {
            antenna_array[i] -= smallest;
        }

        antenna_array[i] *= fs / propagation_speed;
    }

    double epsilon = 1e-9; // Small number to avoid dividing by 0

    // Give each element it's own set of coefficients
    for (int element = 0; element < rows * columns; element++)
    {
        double sum = 0.0;

        // This is the crucial math
        double tau = 0.5 - (double)antenna_array[element] + epsilon;

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

            coefficients[element][i] = (float)h_i_d;
        }

        for (int i = 0; i < N_TAPS; i++)
        {
            // Normalize to get unity gain.
            coefficients[element][i] /= (float)sum;
        }
    }

    free(antenna_array);
}

/*
Python wrapper used to initate the antenna
*/
void py_antenna_wrapper(double azimuth,   // Horizontal
                double elevation, // Vertical
                int columns,
                int rows,
                float distance, // Distance between elements
                float fs,
                float propagation_speed,
                float *coefficients)
{

    float **tmp_coefficients = (float **)malloc((columns * rows) * sizeof(float *));

    for (int i = 0; i < columns * rows; i++)
    {
        tmp_coefficients[i] = (float *)malloc(N_TAPS * sizeof(float));
    }

    directional_antenna_delay_coefficients(azimuth,   // Horizontal
                                           elevation, // Vertical
                                           columns,
                                           rows,
                                           distance, // Distance between elements
                                           fs,
                                           propagation_speed,
                                           tmp_coefficients);

    for (int element = 0; element < columns * rows; element++)
    {
        for (int tap = 0; tap < N_TAPS; tap++)
        {
            coefficients[element * N_TAPS + tap] = tmp_coefficients[element][tap];
        }
    }

    for (int i = 0; i < columns * rows; i++)
    {
        free(tmp_coefficients[i]);
    }

    free(tmp_coefficients);
}

