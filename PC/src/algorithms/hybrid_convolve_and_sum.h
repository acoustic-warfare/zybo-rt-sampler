#ifndef _HYBRID_CONVOLVE_AND_SUM_H
#define _HYBRID_CONVOLVE_AND_SUM_H

void convolve_hybrid_delay_add(float *signal, float *h, int pad, float *out);

void miso_convolve_hybrid(float *signals, float *out, int *adaptive_array, int n, int offset);

void mimo_convolve_hybrid(float *signals, float *image, int *adaptive_array, int n);

void load_coefficients_convolve_hybrid(float *h, int n);

void unload_coefficients_convolve_hybrid();

#endif