#if 0
#include <stdio.h>
#include <time.h>
#define N 10000

// gcc -o run lerp_and_sum.c -lm && ./run

// int main(int argc, char const *argv[])
// {
//     /* code */

//     float signal[N_SAMPLES], out[N_SAMPLES];
//     for (int i = 0; i < N_SAMPLES; i++)
//     {
//         signal[i] = (float)i / 3;
//         out[i] = 0.0;
//     }

//     clock_t tic = clock();
//     for (int i = 0; i < N; i++)
//     {
//         lerp_delay(&signal[0], &out[0], 1 - 0.00001, 0);
//     }

//     // memset(&out[0], 0, N_SAMPLES * sizeof(float));

//     lerp_delay(&signal[0], &out[0], 1 - 0.00001, 0);
//     clock_t toc = clock();
//     printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

//     printf("\n");
//     for (int i = 0; i < 10; i++)
//     {
//         printf("%f %f \n",signal[i], out[i]);
//     }

//     load_coefficients_lerp(&signal[0], N_SAMPLES);
//     for (int i = 0; i < 20; i++)
//     {
//         printf("%d %f\n", whole_samples_lerp[i], fractional_samples_lerp[i]);
//     }

//     unload_coefficients_lerp();

//     return 0;
// }

#include "config.h"
#include "convolve_and_sum.h"

#include "lerp_and_sum.h"

double benchmark_lerp(int n)
{
    float signals[BUFFER_LENGTH];
    for (int i = 0; i < BUFFER_LENGTH; i++)
    {
        signals[i] = (float)i / 100.0;
    }

    float image[MAX_RES_X * MAX_RES_Y];

    int n_h = MAX_RES_X * MAX_RES_Y * N_MICROPHONES;

    float coefficients[MAX_RES_X * MAX_RES_Y * N_MICROPHONES];
    for (int i = 0; i < MAX_RES_X * MAX_RES_Y * N_MICROPHONES; i++)
    {
        coefficients[i] = 1.0 + (float)i / MAX_RES_X * MAX_RES_Y * N_MICROPHONES;
    }

    int adaptive_array[N_MICROPHONES];

    for (int i = 0; i < N_MICROPHONES; i++)
    {
        adaptive_array[i] = i;
    }

    load_coefficients_lerp(&coefficients[0], MAX_RES_X * MAX_RES_Y * N_MICROPHONES);

    clock_t tic = clock();
    for (int i = 0; i < N; i++)
    {
        mimo_lerp(&signals[0], &image[0], &adaptive_array[0], N_MICROPHONES);
    }

    clock_t toc = clock();

    unload_coefficients_lerp();

    return (double)(toc - tic) / CLOCKS_PER_SEC;
}

#define TEST_SIZE 10

int main(int argc, char const *argv[])
{

    double time_lerp = benchmark_lerp(TEST_SIZE);
    printf("Elapsed: %f seconds\n", time_lerp);
    printf("Frame: %f seconds\n", time_lerp / (double)N);
    
    return 0;
}

#endif