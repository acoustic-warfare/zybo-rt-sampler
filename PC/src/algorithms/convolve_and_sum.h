#ifndef _CONVOLVE_AND_SUM_H
#define _CONVOLVE_AND_SUM_H

void convolve_delay_naive_add(float *signal, float *h, float *out);

void convolve_delay_vectorized(float *signal, float *h, float *out);

void convolve_delay_vectorized_add(float *signal, float *h, float *out);

void convolve_delay_naive(float *signal, float *out, float *h);

void convolve_naive(float *signals, float *out, int *adaptive_array, int n, int offset);

void mimo_convolve_naive(float *signals, float *image, int *adaptive_array, int n);

void miso_convolve_vectorized(float *signals, float *out, int *adaptive_array, int n, int offset);

void mimo_convolve_vectorized(float *signals, float *image, int *adaptive_array, int n);

void load_coefficients_convolve(float *h, int n);

void unload_coefficients_convolve();

#endif