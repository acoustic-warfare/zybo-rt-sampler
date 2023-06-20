/******************************************************************************
 * Title                 :   Store Contigiuous Data
 * Filename              :   circular_buffer.h
 * Author                :   Irreq
 * Origin Date           :   120/06/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 9.5.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 Functions to store scalar data efficiently in a circular buffer.

*/

#include "config.h"



typedef struct _ring_buffer
{
    int index;
    float data[BUFFER_LENGTH];
} ring_buffer;


void write_buffer(ring_buffer *rb, float *in, int length, int offset);

void read_buffer(ring_buffer *rb, float *out, int length, int offset);

void read_buffer_mcpy(ring_buffer *rb, float *out);