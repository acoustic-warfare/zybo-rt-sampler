#include <stdlib.h>
#include <string.h> // memcpy

#include "config.h"
#include "circular_buffer.h"

/*

USAGE:

ring_buffer *rb = (ring_buffer *)calloc(1, sizeof(ring_buffer));
rb->index = 0:

write_buffer(rb, ptr, N, 0);

read_buffer_mcpy(rb, &out[0]);

*/

// typedef struct _ring_buffer
// {
//     int index;
//     float data[BUFFER_LENGTH];
// } ring_buffer;

/**
Write data from an address `in` to a ring buffer you can specify offset
but most of the times, it will probably just be 0
*/
void write_buffer(ring_buffer *rb, float *in, int length, int offset)
{
    // -1 for our binary modulo in a moment
    int buffer_length = BUFFER_LENGTH - 1;
    int previous_item = rb->index;

    int idx;
    for (int i = 0; i < length; ++i)
    {
        // modulo will automagically wrap around our index
        idx = (i + previous_item) & buffer_length;
        rb->data[idx] = in[i + offset];
    }

    // Update the current index of our ring buffer.
    rb->index += length;
    rb->index &= BUFFER_LENGTH - 1;
}

/*
Retrieve the data in the ring buffer. You can specify the offset.
the length is the number of previous samples in descending order that will be put into out
on the offset location and onwards.
This may result in stack smashing if offset yields in out of bounds array
*/
void read_buffer(ring_buffer *rb, float *out, int length, int offset)
{
    int index = 0;
    for (int i = rb->index; i < BUFFER_LENGTH; i++)
    {
        if (BUFFER_LENGTH - length <= index)
        {
            out[index + offset - (BUFFER_LENGTH - length)] = rb->data[i];
        }

        index++;
    }

    for (int i = 0; i < rb->index; i++)
    {
        if (BUFFER_LENGTH - length <= index)
        {
            out[index + offset - (BUFFER_LENGTH - length)] = rb->data[i];
        }

        index++;
    }
}

/*
Almost 100 times faster than the code above
*/
void read_buffer_mcpy(ring_buffer *rb, float *out)
{
    int first_partition = BUFFER_LENGTH - rb->index;

    float *data_ptr = &rb->data[0];

    memcpy(out, (void *)(data_ptr + rb->index), sizeof(float) * first_partition);
    memcpy(out + first_partition, (void *)(data_ptr), sizeof(float) * rb->index);
}
