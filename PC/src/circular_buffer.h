/******************************************************************************
 * Title                 :   Store Contigiuous Data
 * Filename              :   circular_buffer.h
 * Author                :   Irreq
 * Origin Date           :   20/06/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 9.5.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 Functions to store scalar data efficiently in a circular buffer.

*/

#ifndef _CIRCULAR_BUFFER_H_
#define _CIRCULAR_BUFFER_H_

#include "config.h"



typedef struct _ring_buffer
{
    int index;
    float data[BUFFER_LENGTH];
} ring_buffer;

struct ringba
{
    int index;
    float data[BUFFER_LENGTH];
};

ring_buffer *create_ring_buffer();

ring_buffer *destroy_ring_buffer(ring_buffer *rb);

void write_buffer(ring_buffer *rb, float *in, int length, int offset);

void write_buffer_int32(ring_buffer *rb, int32_t *in, int length, int offset);

void read_buffer(ring_buffer *rb, float *out, int length, int offset);

void read_buffer_mcpy(ring_buffer *rb, float *out);

void write_int32(struct ringba *rb, int32_t *in, int length, int offset);

void read_mcpy(struct ringba *rb, float *out);

#endif