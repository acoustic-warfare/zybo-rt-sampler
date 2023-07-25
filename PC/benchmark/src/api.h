#ifndef _API_H
#define _API_H

#include <stdbool.h>

int load(bool replay_mode);
void get_data(float *signals);
void stop_receiving();

void pad_mimo(float *image, int *adaptive_array, int n);
void convolve_mimo_naive(float *image, int *adaptive_array, int n);
void convolve_mimo_vectorized(float *image, int *adaptive_array, int n);


void mimo_truncated(float *image, int *adaptive_array, int n);
void load_coefficients2(int *whole_samples, int n);
#endif