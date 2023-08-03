#ifndef _LERP_AND_SUM_H
#define _LERP_AND_SUM_H

void lerp_delay(float *signal, float *out, float h, int pad);

void miso_lerp(float *signals, float *out, int *adaptive_array, int n, int offset);

void load_coefficients_lerp(float *delays, int n);

void unload_coefficients_lerp();

#endif