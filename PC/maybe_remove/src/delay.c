#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <immintrin.h>

// #include "delay.h"

#define N_TAPS 8 // 256 // 16
#define OFFSET N_TAPS / 2

#define PI 3.14159265359

#define N_SAMPLES 64 // 64 // 1024 // 128 // 65536 // 2^16
#define ALIGNMENT 32

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

/*
Delay a signal by creating a reversed convolution of size (N + M)
and return only of size N using the immintrin AVX instructions
using unrolled Fused Multiply Addition (FMA)
*/
void delay_vectorized(float *signal, float *h, float *out)
{
    // Allign data
    __m256 aligned_kernel[KERNEL_LENGTH] __attribute__((aligned(ALIGNMENT)));
    __m256 data_block __attribute__((aligned(ALIGNMENT)));

    // Two accumulators are set up
    __m256 accumulator0 __attribute__((aligned(ALIGNMENT)));
    __m256 accumulator1 __attribute__((aligned(ALIGNMENT)));

    // Repeat the kernel across the vector
    for (int i = 0; i < KERNEL_LENGTH; i++)
    {
        aligned_kernel[i] = _mm256_broadcast_ss(&h[i]);
    }

    // Allocate memory for the padded signal
    float *padded = malloc((N_SAMPLES + N_TAPS) * sizeof(float));

    // Zero entire buffer
    for (int i = 0; i < N_SAMPLES + N_TAPS; i++)
    {
        padded[i] = 0.0;
    }

    // Load signal to buffer
    for (int i = 0; i < N_SAMPLES; i++)
    {
        padded[i + OFFSET] = signal[i];
    }

    for (int i = 0; i < N_SAMPLES; i += VECTOR_LENGTH)
    {
        accumulator0 = _mm256_setzero_ps();
        accumulator1 = _mm256_setzero_ps();

        for (int k = 0; k < KERNEL_LENGTH; k += VECTOR_LENGTH)
        {
            int data_offset = i + k;

            for (int l = 0; l < SSE_SIMD_LENGTH; l++)
            {

                for (int m = 0; m < VECTOR_LENGTH; m += SSE_SIMD_LENGTH)
                {
                    // First block
                    data_block = _mm256_loadu_ps(padded + l + data_offset + m);

                    accumulator0 = _mm256_fmadd_ps(
                        aligned_kernel[k + l + m], data_block, accumulator0);

                    // Second block
                    data_block = _mm256_loadu_ps(padded + l + data_offset + m + AVX_SIMD_LENGTH);

                    accumulator1 = _mm256_fmadd_ps(
                        aligned_kernel[k + l + m], data_block, accumulator1);
                }
            }
        }
        _mm256_storeu_ps(out + i, accumulator0);
        _mm256_storeu_ps(out + i + AVX_SIMD_LENGTH, accumulator1);
    }

    // Need to do the last value as a special case
    int i = N_SAMPLES - 1;
    out[i] = 0.0;
    for (int k = 0; k < KERNEL_LENGTH; k++)
    {
        out[i] += h[k] * padded[i + k];
    }

    free(padded);
}

//int convolve_avx(float *in, int input_length,
//                 float *kernel, int kernel_length, float *out)
//{
//    float *in_padded = (float *)(_alloca(sizeof(float) * (input_length + 16)));
//
//    __m256 *kernel_many = (__m256 *)(_alloca(sizeof(__m256) * kernel_length));
//
//    __m256 block;
//    __m256 prod;
//    __m256 acc;
//
//    // Repeat the kernel across the vector
//    int i;
//    for (i = 0; i < kernel_length; i++)
//    {
//        kernel_many[i] = _mm256_broadcast_ss(&kernel[i]); // broadcast
//    }
//
//    /* Create a set of 4 aligned arrays
//     * Each array is offset by one sample from the one before
//     */
//    block = _mm256_setzero_ps();
//    _mm256_storeu_ps(in_padded, block);
//    memcpy(&(in_padded[8]), in, input_length * sizeof(float));
//    _mm256_storeu_ps(in_padded + input_length + 8, block);
//
//    for (i = 0; i < input_length + kernel_length - 8; i += 8)
//    {
//
//        // Zero the accumulator
//        acc = _mm256_setzero_ps();
//
//        int startk = i > (input_length - 1) ? i - (input_length - 1) : 0;
//        int endk = (i + 7) < kernel_length ? (i + 7) : kernel_length - 1;
//
//        int k = startk;
//
//        // Manual unrolling of the loop to trigger pipelining speed-up (x2 perf)
//        for (; k + 3 <= endk; k += 4)
//        {
//            block = _mm256_loadu_ps(in_padded + 8 + i - k);
//            prod = _mm256_mul_ps(block, kernel_many[k]);
//            acc = _mm256_add_ps(acc, prod);
//
//            block = _mm256_loadu_ps(in_padded + 7 + i - k);
//            prod = _mm256_mul_ps(block, kernel_many[k + 1]);
//            acc = _mm256_add_ps(acc, prod);
//
//            block = _mm256_loadu_ps(in_padded + 6 + i - k);
//            prod = _mm256_mul_ps(block, kernel_many[k + 2]);
//            acc = _mm256_add_ps(acc, prod);
//
//            block = _mm256_loadu_ps(in_padded + 5 + i - k);
//            prod = _mm256_mul_ps(block, kernel_many[k + 3]);
//            acc = _mm256_add_ps(acc, prod);
//        }
//
//        for (; k <= endk; k++)
//        {
//            block = _mm256_loadu_ps(in_padded + 8 + i - k);
//            prod = _mm256_mul_ps(block, kernel_many[k]);
//            acc = _mm256_add_ps(acc, prod);
//        }
//
//        _mm256_storeu_ps(out + i, acc);
//    }
//
//    // Left-overs
//    for (; i < input_length + kernel_length - 1; i++)
//    {
//
//        out[i] = 0.0;
//        int startk = i >= input_length ? i - input_length + 1 : 0;
//        int endk = i < kernel_length ? i : kernel_length - 1;
//        for (int k = startk; k <= endk; k++)
//        {
//            out[i] += in[i - k] * kernel[k];
//        }
//    }
//
//    return 0;
//}

void delay_vectorized2(float *signal, float *h, float *out)
{
    // Allign data
    __m256 aligned_kernel[N_TAPS / AVX_SIMD_LENGTH] __attribute__((aligned(ALIGNMENT)));
    __m256 data_block __attribute__((aligned(ALIGNMENT)));
    __m256 kernel_block __attribute__((aligned(ALIGNMENT)));

    // // Allocate memory for the padded signal with proper alignment
    // float *padded = (float *)_mm_malloc((N_SAMPLES + N_TAPS) * sizeof(float), ALIGNMENT);

    // // Zero entire buffer
    // for (int i = 0; i < N_SAMPLES + N_TAPS; i++)
    // {
    //     padded[i] = 0.0;
    // }

    // // Load signal to buffer
    // for (int i = 0; i < N_SAMPLES; i++)
    // {
    //     padded[i + OFFSET] = signal[i];
    // }

    float padded[N_SAMPLES + N_TAPS];

    // Initialize the padded array with zeros
    memset(padded, 0, (N_SAMPLES + N_TAPS) * sizeof(float));

    // Copy the original array to the padded array
    memcpy(padded + OFFSET, signal, N_SAMPLES * sizeof(float));

    // Repeat the kernel across the vector
    for (int i = 0; i < N_TAPS / AVX_SIMD_LENGTH; i++)
    {
        aligned_kernel[i] = _mm256_broadcast_ss(&h[i * AVX_SIMD_LENGTH]);
    }

    // Two accumulators are set up
    __m256 accumulator0 __attribute__((aligned(ALIGNMENT)));
    __m256 accumulator1 __attribute__((aligned(ALIGNMENT)));
    __m256 accumulator2 __attribute__((aligned(ALIGNMENT)));
    __m256 accumulator3 __attribute__((aligned(ALIGNMENT)));

    for (int i = 0; i < N_SAMPLES; i += VECTOR_LENGTH * 2)
    {
        accumulator0 = _mm256_setzero_ps();
        accumulator1 = _mm256_setzero_ps();

        accumulator2 = _mm256_setzero_ps();
        accumulator3 = _mm256_setzero_ps();

        data_block = _mm256_load_ps(padded + i);
        accumulator0 = _mm256_fmadd_ps(data_block, aligned_kernel[0], accumulator0);

        data_block = _mm256_load_ps(padded + i + AVX_SIMD_LENGTH);
        accumulator1 = _mm256_fmadd_ps(data_block, aligned_kernel[1], accumulator1);

        data_block = _mm256_load_ps(padded + i + AVX_SIMD_LENGTH * 2);
        accumulator2 = _mm256_fmadd_ps(data_block, aligned_kernel[2], accumulator2);

        data_block = _mm256_load_ps(padded + i + AVX_SIMD_LENGTH * 3);
        accumulator3 = _mm256_fmadd_ps(data_block, aligned_kernel[3], accumulator3);

        _mm256_storeu_ps(out + i, accumulator0);
        _mm256_storeu_ps(out + i, accumulator1);
        _mm256_storeu_ps(out + i, accumulator2);
        _mm256_storeu_ps(out + i, accumulator3);
    }
}

void dumb_vec(float *signal, float *h, float *out)
{
    //__builtin_prefetch(&h[i + PD], WILL_READ_ONLY, LOCALITY_LOW);
}

void delay_vectorized5(float *signal, float *h, float *out)
{
    // Allign data
    __m256 aligned_kernel[N_TAPS / AVX_SIMD_LENGTH] __attribute__((aligned(ALIGNMENT)));
    __m256 data_block __attribute__((aligned(ALIGNMENT)));
    __m256 kernel_block __attribute__((aligned(ALIGNMENT)));

    //float *padded[N_SAMPLES + N_TAPS];
    //// Initialize the padded array with zeros
    //memset(padded, 0, (N_SAMPLES + N_TAPS) * sizeof(float));
    //// Copy the original array to the padded array
    //memcpy(padded + OFFSET, signal, N_SAMPLES * sizeof(float));

    // Allocate memory for the padded signal
    float *padded = malloc((N_SAMPLES + N_TAPS) * sizeof(float));

    // Zero entire buffer
    for (int i = 0; i < N_SAMPLES + N_TAPS; i++)
    {
        padded[i] = 0.0;
    }

    // Load signal to buffer
    for (int i = 0; i < N_SAMPLES; i++)
    {
        padded[i + OFFSET] = signal[i];
    }


    // Repeat the kernel across the vector
    // for (int i = 0; i < N_TAPS / AVX_SIMD_LENGTH; i++)
    // {
    //     aligned_kernel[i] = _mm256_broadcast_ss(&h[i * AVX_SIMD_LENGTH]);
    // }
    __m256 accumulator __attribute__((aligned(ALIGNMENT)));
    for (int i = 0; i < N_SAMPLES; i += 1)
    {
        accumulator = _mm256_setzero_ps();

        data_block = _mm256_load_ps(padded + i);
        kernel_block = _mm256_load_ps(h);
        accumulator = _mm256_fmadd_ps(data_block, kernel_block, accumulator);

        //for (int k = 0; k < N_TAPS / AVX_SIMD_LENGTH; k++)
        //{
        //    //printf("%d ", (int)k);
        //    data_block = _mm256_load_ps(&padded[0] + i + AVX_SIMD_LENGTH * k);
        //    accumulator = _mm256_fmadd_ps(data_block, aligned_kernel[k], accumulator);
        //    
        //}
//
        //for (size_t i = 0; i < 3; i++)
        //{
        //    printf("%f ", accumulator[i]);
        //}
        

        out[i] = hsum256_ps_avx(accumulator);

        // printf("%f\n", hsum256_ps_avx(accumulator));

        //_mm256_storeu_ps(out + i, accumulator);
    }

    free(padded);
}

void delay_vectorized6(float *signal, float *h, float *out)
{
    // Allign data
    __m256 aligned_kernel[2] __attribute__((aligned(ALIGNMENT)));
    __m256 data_block __attribute__((aligned(ALIGNMENT)));
    __m256 kernel_block __attribute__((aligned(ALIGNMENT)));

    float padded[N_SAMPLES + N_TAPS];

    // Initialize the padded array with zeros
    memset(padded, 0, (N_SAMPLES + N_TAPS) * sizeof(float));

    // Copy the original array to the padded array
    memcpy(padded + OFFSET, signal, N_SAMPLES * sizeof(float));

    for (size_t i = 0; i < N_SAMPLES; i++)
    {
        printf("%f ", padded[i]);
    }

    // Repeat the kernel across the vector
    for (int i = 0; i < 2; i++)
    {
        aligned_kernel[i] = _mm256_broadcast_ss(&h[i * AVX_SIMD_LENGTH]);
    }

    __m256 accumulator0 __attribute__((aligned(ALIGNMENT)));
    __m256 accumulator1 __attribute__((aligned(ALIGNMENT)));

    for (int i = 0; i < N_SAMPLES; i += 1)
    {
        accumulator0 = _mm256_setzero_ps();
        // accumulator1 = _mm256_setzero_ps();

        data_block = _mm256_load_ps(padded + i);
        accumulator0 = _mm256_fmadd_ps(data_block, aligned_kernel[0], accumulator0);

        // data_block = _mm256_load_ps(padded + i + AVX_SIMD_LENGTH);
        // accumulator1 = _mm256_fmadd_ps(data_block, aligned_kernel[1], accumulator1);
        out[i] = hsum256_ps_avx(accumulator0);
            // _mm256_storeu_ps(out + i, accumulator0);
        // _mm256_storeu_ps(out + i, accumulator1);
    }
}

void delay_vectorized4(float *signals, float *kernels, float *result, int num_signals, int signal_length, int kernel_length)
{
    // Calculate the remaining length after AVX vectorization
    int remainder = signal_length % 8;
    int vectorized_length = signal_length - remainder;

    // Loop over each signal
    for (int s = 0; s < num_signals; s++)
    {
        // Loop over each element in the result
        for (int i = 0; i < signal_length - kernel_length + 1; i++)
        {
            // Initialize the accumulator
            __m256 sum = _mm256_setzero_ps();

            // Loop over each kernel element
            for (int j = 0; j < kernel_length; j += 8)
            {
                // Load 8 signals elements at a time
                __m256 signal_vec = _mm256_loadu_ps(signals + s * signal_length + i + j);

                // Load 8 kernel elements at a time
                __m256 kernel_vec = _mm256_loadu_ps(kernels + s * kernel_length + j);

                // Perform the element-wise multiplication
                __m256 mul_result = _mm256_mul_ps(signal_vec, kernel_vec);

                // Accumulate the results
                sum = _mm256_add_ps(sum, mul_result);
            }

            // Reduce the sum across the 8 elements using horizontal add
            sum = _mm256_hadd_ps(sum, sum);
            sum = _mm256_hadd_ps(sum, sum);

            // Store the result
            _mm_store_ss(result + s * (signal_length - kernel_length + 1) + i, _mm256_castps256_ps128(sum));
        }

        // Process the remaining elements using scalar code
        for (int i = vectorized_length; i < signal_length - kernel_length + 1; i++)
        {
            float sum = 0.0f;

            // Loop over each kernel element
            for (int j = 0; j < kernel_length; j++)
            {
                sum += signals[s * signal_length + i + j] * kernels[s * kernel_length + j];
            }

            result[s * (signal_length - kernel_length + 1) + i] = sum;
        }
    }
}

void delay_vectorized3(float *signal, float *h, float *out)
{
    // Align data
    __m256 aligned_kernel[KERNEL_LENGTH] __attribute__((aligned(ALIGNMENT)));
    __m256 data_block, kernel_block;

    // Allocate memory for the padded signal with proper alignment
    float *padded = (float *)_mm_malloc((N_SAMPLES + N_TAPS) * sizeof(float), ALIGNMENT);

    // Zero entire buffer
    for (int i = 0; i < N_SAMPLES + N_TAPS; i++)
    {
        padded[i] = 0.0;
    }

    // Load signal to buffer
    for (int i = 0; i < N_SAMPLES; i++)
    {
        padded[i + OFFSET] = signal[i];
    }

    // Repeat the kernel across the vector
    for (int i = 0; i < KERNEL_LENGTH; i++)
    {
        aligned_kernel[i] = _mm256_broadcast_ss(&h[i]);
    }

    // Single accumulator is set up
    __m256 accumulator __attribute__((aligned(ALIGNMENT)));

    for (int i = 0; i < N_SAMPLES; i += VECTOR_LENGTH)
    {
        accumulator = _mm256_setzero_ps();

        data_block = _mm256_load_ps(padded + i);
        accumulator = _mm256_fmadd_ps(data_block, aligned_kernel[0], accumulator);

        data_block = _mm256_load_ps(padded + i + AVX_SIMD_LENGTH);
        accumulator = _mm256_fmadd_ps(data_block, aligned_kernel[1], accumulator);

        // Vertical summation
        accumulator = _mm256_hadd_ps(accumulator, accumulator);
        accumulator = _mm256_hadd_ps(accumulator, accumulator);

        // Store the result
        _mm256_storeu_ps(out + i, accumulator);
    }

    // Free the allocated memory
    _mm_free(padded);
}

// /*
// Calculate the different coefficients depeding on the antenna's direction and size
// */
// void directional_antenna_delay_coefficients(double azimuth,   // Horizontal
//                                             double elevation, // Vertical
//                                             int columns,
//                                             int rows,
//                                             float distance, // Distance between elements
//                                             float fs,
//                                             float propagation_speed,
//                                             float **coefficients)

// {
//     // Convert listen direction to radians
//     double theta = azimuth * -(double)PI / 180.0;
//     double phi = elevation * -(double)PI / 180.0;

//     float x_factor = (float)(sin(theta) * cos(phi));
//     float y_factor = (float)(cos(theta) * sin(phi));

//     // Allocate antenna array
//     float *antenna_array = malloc((columns * rows) * sizeof(float));

//     int element_index = 0;

//     float smallest = 0.0;

//     // Create antenna in space with middle in origo (0, 0)
//     for (int row = 0; row < rows; row++)
//     {
//         for (int col = 0; col < columns; col++)
//         {
//             float half = distance / 2.0;

//             // Assign middle of array to origo
//             float tmp_col = (float)col * distance - (float)columns * half + half;
//             float tmp_row = (float)row * distance - (float)rows    * half + half;

//             float tmp_delay = tmp_col * x_factor + tmp_row * y_factor;

//             // Update so there is always one element furthest from world at 0 i.e all other delays are greater
//             if (tmp_delay < smallest)
//             {
//                 smallest = tmp_delay;
//             }

//             antenna_array[element_index] = tmp_delay;

//             element_index += 1;
//         }
//     }

//     // Create a delay map
//     for (int i = 0; i < rows * columns; i++)
//     {
//         // Make the furthest element from source direction have no delay
//         if (smallest < 0.0)
//         {
//             antenna_array[i] -= smallest;
//         }

//         antenna_array[i] *= fs / propagation_speed;
//     }

//     double epsilon = 1e-9;  // Small number to avoid dividing by 0

//     // Give each element it's own set of coefficients
//     for (int element = 0; element < rows * columns; element++)
//     {
//         double sum = 0.0;

//         // This is the crucial math
//         double tau = 0.5 - (double)antenna_array[element] + epsilon;

//         for (int i = 0; i < N_TAPS; i++)
//         {
//             // Fractional delay with support to delay entire frames up to OFFSET
//             double h_i_d = (double)i - ((double)N_TAPS - 1.0) / 2.0 - tau;
//             // Compute the sinc value: sin(xπ)/xπ
//             h_i_d = sin(h_i_d * PI) / (h_i_d * PI);

//             // To get np.arange(1-M, M, 2)
//             double n = (double)(i * 2 - N_TAPS + 1);

//             // Multiply sinc value by Blackman-window (https://numpy.org/doc/stable/reference/generated/numpy.blackman.html)
//             double black_manning = 0.42 + 0.5 * cos(PI * n / ((double)(N_TAPS - 1)) + epsilon) + 0.08 * cos(2.0 * PI * n / ((double)(N_TAPS - 1) + epsilon));

//             h_i_d *= black_manning;

//             sum += h_i_d;

//             coefficients[element][i] = (float)h_i_d;
//         }

//         for (int i = 0; i < N_TAPS; i++)
//         {
//             // Normalize to get unity gain.
//             coefficients[element][i] /= (float)sum;
//         }
//     }

//     free(antenna_array);
// }


int convolve_avx_unrolled_vector_unaligned_fma(float *in, float *out,
                                                   int length, float *kernel, int kernel_length)
{

    __m256 kernel_reverse[KERNEL_LENGTH] __attribute__((aligned(ALIGNMENT)));
    __m256 data_block __attribute__((aligned(ALIGNMENT)));

    __m256 acc0 __attribute__((aligned(ALIGNMENT)));
    __m256 acc1 __attribute__((aligned(ALIGNMENT)));

    // Repeat the kernel across the vector
    for (int i = 0; i < KERNEL_LENGTH; i++)
    {
        kernel_reverse[i] = _mm256_broadcast_ss(
            &kernel[i]);
    }

    for (int i = 0; i < length - KERNEL_LENGTH; i += VECTOR_LENGTH)
    {

        acc0 = _mm256_setzero_ps();
        acc1 = _mm256_setzero_ps();

        for (int k = 0; k < KERNEL_LENGTH; k += VECTOR_LENGTH)
        {

            int data_offset = i + k;

            for (int l = 0; l < SSE_SIMD_LENGTH; l++)
            {

                for (int m = 0; m < VECTOR_LENGTH; m += SSE_SIMD_LENGTH)
                {

                    data_block = _mm256_loadu_ps(
                        in + l + data_offset + m);

                    // acc0 = kernel_reverse[k+l+m] * data_block + acc0;
                    acc0 = _mm256_fmadd_ps(
                        kernel_reverse[k + l + m], data_block, acc0);

                    data_block = _mm256_loadu_ps(in + l + data_offset + m + AVX_SIMD_LENGTH);

                    // acc1 = kernel_reverse[k+l+m] * data_block + acc1;
                    acc1 = _mm256_fmadd_ps(
                        kernel_reverse[k + l + m], data_block, acc1);
                }
            }
        }
        _mm256_storeu_ps(out + i, acc0);
        _mm256_storeu_ps(out + i + AVX_SIMD_LENGTH, acc1);
    }

    // Need to do the last value as a special case
    int i = length - KERNEL_LENGTH;
    out[i] = 0.0;
    for (int k = 0; k < KERNEL_LENGTH; k++)
    {
        out[i] += in[i + k] * kernel[KERNEL_LENGTH - k - 1];
    }

    return 0;
}

int convolve_naive(float *in, float *out, int length,
                   float *kernel, int kernel_length)
{
    for (int i = 0; i <= length - kernel_length; i++)
    {

        out[i] = 0.0;
        for (int k = 0; k < kernel_length; k++)
        {
            out[i] += in[i + k] * kernel[kernel_length - k - 1];
        }
    }

    return 0;
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

int main()
{
    float *h = malloc(N_TAPS * sizeof(float));

    float *signal = malloc(N_SAMPLES * sizeof(float));

    generate_coefficients(0.5, h);

    for (int i = 0; i < N_SAMPLES; i++)
    {
        signal[i] = (float)i; // / (float)N_SAMPLES;
    }

    // float out[N_SAMPLES];

    // // Copy zeros into the array using memset
    // memset(out, 0, sizeof(out));

    // delay_naive(signal, h, &out[0]);

    int test_samples = 10000;

    float *out = (float *)aligned_alloc(ALIGNMENT, N_SAMPLES * sizeof(float));

    // // Perform convolution using vectorized implementation
    // clock_t start = clock();
    // for (int i = 0; i < test_samples; i++)
    // {
    //     delay_vectorized5(signal, h, out);
    //     // vectorizedConvolution(signals, kernels, result_vectorized, num_signals, signal_length, kernel_length);
    // }
    // clock_t end = clock();
    // double vectorized_time = (double)(end - start) / CLOCKS_PER_SEC;// / test_samples;

    // // Perform convolution using naive implementation
    // start = clock();
    // for (int i = 0; i < test_samples; i++)
    // {
    //     delay_naive(signal, h, out);
    //     // delay_vectorized(signal, h, out);
    //     // naiveConvolution(signals, kernels, result_naive, num_signals, signal_length, kernel_length);
    // }
    // end = clock();

    // double naive_time = (double)(end - start) / CLOCKS_PER_SEC;// / test_samples;

    // printf("Vectorized Convolution Execution Time: %f seconds\n", vectorized_time);
    // printf("Naive Convolution Execution Time: %f seconds\n", naive_time);

    delay_vectorized5(signal, h, out);

    //convolve_avx_unrolled_vector_unaligned_fma(signal, out, N_SAMPLES, h, N_TAPS);
    //convolve_naive(signal, out, N_SAMPLES, h, N_TAPS);

    float *out2 = (float *)aligned_alloc(ALIGNMENT, N_SAMPLES * sizeof(float));

    delay_naive(signal, h, out2);

    for (int i = 0; i < N_SAMPLES; i++)
    {
        printf("%f %f %f\n", signal[i], out[i], out2[i]);
    }

    free(h);
    free(signal);
}