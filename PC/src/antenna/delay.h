#ifndef _DELAY_H_
#define _DELAY_H_


#include <immintrin.h>

void delay_vectorized_add(float *signal, float *h, float *out);

float sum8(__m256 x);

#endif