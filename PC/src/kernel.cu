#include "kernel.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "filter_coefficients.h"


#define NX 256
#define BATCH 256
#define RANK 1

cufftHandle plan2;
cufftHandle plan3;
float *cuda_input; 
cufftComplex *inputFFT;
cufftComplex *conv;
cufftComplex *filter_FFT;
float *res;

__global__ void spectral_convolution(cufftComplex *filter, cufftComplex *input, cufftComplex *out, size_t size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    cufftComplex h = filter[tid];
    cufftComplex x = input[tid];

    out[tid].x = (x.x * h.x) - (x.y * h.y);
    out[tid].y = (x.x * h.y) + (x.y * h.x);
}

__global__ void normalize(float *signal){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    signal[tid] = signal[tid] / (256);
}

void cuda_filter_init(){
    size_t filter_size_bytes = NX * sizeof(float) * 256;
    size_t filter_fft_size_bytes = 129 *sizeof(cufftComplex) * BATCH;
    
    float *cuda_filter_coefficients;
    
    cudaMalloc((void **) &cuda_filter_coefficients, filter_size_bytes);
    cudaMalloc((void **) &filter_FFT, filter_fft_size_bytes);
    float *filter_coeffs = H;

    for(int i = 0; i < BATCH; i ++){
        cudaMemcpy((void *) &cuda_filter_coefficients[i*BATCH], (void *) filter_coeffs, 256 * sizeof(float) , cudaMemcpyHostToDevice);
    }
    
    cufftHandle plan_filter_fft;
    cufftResult status = cufftCreate(&plan_filter_fft);
    if(status != 0){
        printf("CUDA_ERROR %d", status);
    }

    status = cufftPlan1d(&plan_filter_fft, NX, CUFFT_R2C, BATCH);
    if(status != 0){
        printf("CUDA_ERROR %d", status);
    }

    status = cufftExecR2C(plan_filter_fft, cuda_filter_coefficients, filter_FFT);
    if(status != 0){
        printf("CUDA_ERROR %d", status);
    }
    
    

    cudaFree(cuda_filter_coefficients);
    cufftDestroy(plan_filter_fft);
}

void plan_init(){
    cufftResult status = cufftCreate(&plan3);
    if(status != 0){
        printf("CUDA_ERROR_1 %d", status);
    }
    status = cufftPlan1d(&plan3, NX, CUFFT_C2R, BATCH);
    if(status != 0){
        printf("CUDA_ERROR_2 %d", status);
    }

    status = cufftCreate(&plan2);
    if(status != 0){
        printf("CUDA_ERROR_3 %d", status);
    }

    status = cufftPlan1d(&plan2, NX, CUFFT_R2C, BATCH);
    if(status != 0){
        printf("CUDA_ERROR_4 %d", status);
    }
}

void memory_init(){
    size_t input_size_bytes = NX * sizeof(float) *BATCH;
    size_t output_size_bytes = 129 * sizeof(cufftComplex) * BATCH;
    //float *c_input;
    cudaMalloc((void **) &cuda_input, input_size_bytes);
    cudaMalloc((void **) &inputFFT, output_size_bytes);
    cudaMalloc((void **) &conv, output_size_bytes);
    cudaMalloc((void **) &res, input_size_bytes);
}

void cuda_init_all(){
    cuda_filter_init();
    plan_init();
    memory_init();
}

void cuda_filter_FFT(float *input, float *filter_output){
    size_t input_size_bytes = NX * sizeof(float) *BATCH;
    //Init output memory
    cudaMemcpy((void *) cuda_input, (void *) input, input_size_bytes, cudaMemcpyHostToDevice);
    
    //Perform FFT of microphone data
    cufftResult status = cufftExecR2C(plan2, cuda_input, inputFFT);
    if(status != 0){
        printf("CUDA_ERROR_5 %d", status);
    }
    //Perform convolution
    spectral_convolution<<<129,256>>>(filter_FFT, inputFFT, conv, 129);
    
    //Inverse FFT
    status = cufftExecC2R(plan3, conv, res);
    if(status != 0){
        printf("CUDA_ERROR_6 %d", status);
    }
    normalize<<<256, 256>>>(res);
    cudaMemcpy(filter_output, res, input_size_bytes, cudaMemcpyDeviceToHost);
}