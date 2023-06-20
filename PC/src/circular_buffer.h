#include "config.h"

typedef struct _ring_buffer
{
    int index;
    float data[BUFFER_LENGTH];
} ring_buffer;


void write_buffer(ring_buffer *rb, float *in, int length, int offset);

void read_buffer(ring_buffer *rb, float *out, int length, int offset);

void read_buffer_mcpy(ring_buffer *rb, float *out);