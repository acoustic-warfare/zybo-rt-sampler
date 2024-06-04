#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include "kernel.h"
#include "filter_coefficients_CPU.h"

#define INPUT_LENGTH 256
#define FILTER_LENGTH 257
#define BUFFER_LENGTH (INPUT_LENGTH + FILTER_LENGTH - 1)

float insamp[BUFFER_LENGTH];

void init_fir(){
    memset(insamp, 0, sizeof(insamp));
}


void fir_filter(float **coeffs1, float *input, float *output, int length, int filter_length){
    int n;
    int k;
    int i;
    __m256 accum[8];
    float *acc = (float *)_mm_malloc(8 * sizeof(float), 32);

    memcpy(&insamp[0], input, length * sizeof(float));

    for(n = 0; n < length; n+=8){

        for(k = 0; k < 8; k++){
            accum[k] = _mm256_setzero_ps();
        }

        for(k = 0; k < filter_length; k+=8){
            __m256 x = _mm256_load_ps(&insamp[n+k]);
            
            for(i = 0; i < 8; i++){
                __m256 c1 = _mm256_load_ps(&coeffs1[i][k]);  
                
                accum[i] = _mm256_fmadd_ps(x, c1, accum[i]);
            }
        }

        for(k = 0; k < 8; k++){
            _mm256_store_ps(acc, accum[k]);
            float tmp = 0;

            for(int i = 0; i < 8; i++){
                tmp = tmp + acc[i];
            }
            
            output[n+k] = tmp;
        }
    }
    _mm_free(acc);
}

#define SAMPLES 256

float **generate_coeffs_copies_preset(float *rev_coeffs){
    float **c_copies_LUT = (float **)_mm_malloc(8*sizeof(float*), 32);

    for(int i = 0; i < 8; i++){
        c_copies_LUT[i] = (float *) _mm_malloc((256+8) * sizeof(float), 32);
        for(int j = 0; j < 256+8; j++){
            c_copies_LUT[i][j] = 0;
        }
        for(int j = 0; j < 256; j++){
            c_copies_LUT[i][i+j] = rev_coeffs[j];
        }
    }
    return c_copies_LUT;
}

float *filter_one_channel(float *input, float * output, float **coeffs1){
    int size = SAMPLES;
    init_fir();
    fir_filter(coeffs1, input, output, size, size);
    return output;
}

float *reverse_filter_coeffs_merge(){
    float *filter_pointer = H2;

    float *rev = (float *)_mm_malloc(HL2 * sizeof(float), 32);

    for(int i = 0; i < HL2; i++){
        rev[i] = filter_pointer[HL2-1-i];
    }
    

    return rev;
}

//Filter using CUDA
void cuda_filter_rt(float *all_data, float *filter_input, float *filter_output){    
    cuda_filter_FFT(&all_data[0], filter_output);
    memcpy(&all_data[0], &filter_output[0], 256*256*sizeof(float));
}

//Filter using SIMD
float *simd_fir_filter_wma(float** filter_1_coeffs, float* all_data, float * filter_input, float* filter_output){
    
    for(int j = 0; j < 256; j++){
        filter_one_channel(&all_data[j*256], filter_output, filter_1_coeffs);

        memcpy(&all_data[j*256], &filter_output[0], 256*sizeof(float));
    }
    return all_data;
}