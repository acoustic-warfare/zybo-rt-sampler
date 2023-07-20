#ifndef _PAD_AND_SUM_H
#define _PAD_AND_SUM_H


void pad_delay(float *signal, float *out, int pos_pad);
void miso_pad(float *signals, float *out, int *adaptive_array, int n, int offset);
void mimo_pad(float *signals, float *image, int *adaptive_array, int n);

void load_coefficients_pad(int *whole_samples, int n);
void unload_coefficients_pad();

#endif