/******************************************************************************
 * Title                 :   Store Contigiuous Data
 * Filename              :   circular_buffer.c
 * Author                :   Irreq
 * Origin Date           :   20/06/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 9.5.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 Functions to store scalar data efficiently in a circular buffer.

 USAGE:

 #include "circular_buffer.h"

 ring_buffer *rb = create_ring_buffer();

 write_buffer(rb, ptr, N, 0);

 read_buffer_mcpy(rb, &out[0]);

 rb = destroy_ring_buffer(rb);
 
*/

#include <stdlib.h>
#include <string.h> // memcpy
#include <stdint.h>
#include "config.h"
#include "circular_buffer.h"

/*
Create a ring buffer
*/
ring_buffer *create_ring_buffer(){
    ring_buffer *rb = (ring_buffer *)calloc(1, sizeof(ring_buffer));
    rb->index = 0;
    return rb;
}

/*
Destroy a ring buffer
*/
ring_buffer *destroy_ring_buffer(ring_buffer *rb){
    free(rb);
    rb = NULL;
}

/*
Write data from an address `in` to a ring buffer you can specify offset
but most of the times, it will probably just be 0
*/
void write_buffer(ring_buffer *rb, float *in, int length, int offset)
{
    int buffer_length = BUFFER_LENGTH - 1;
    int previous_item = rb->index;

    int idx;
    for (int i = 0; i < length; ++i)
    {
        
        idx = (i + previous_item) & buffer_length; // Wrap around
        rb->data[idx] = in[i + offset];
    }

    // Sync current index
    rb->index += length;
    rb->index &= BUFFER_LENGTH - 1;
}

/*
Write data from an address `in` to a ring buffer you can specify offset
but most of the times, it will probably just be 0
*/
void write_buffer_int32(ring_buffer *rb, int32_t *in, int length, int offset)
{
    int buffer_length = BUFFER_LENGTH - 1;
    int previous_item = rb->index;

    int idx;
    for (int i = 0; i < length; ++i)
    {
        
        idx = (i + previous_item) & buffer_length; // Wrap around
        rb->data[idx] = (float)in[i + offset];
    }

    // Sync current index
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
Almost 100 times faster than the code above but does the same
*/
void read_buffer_mcpy(ring_buffer *rb, float *out)
{
    int first_partition = BUFFER_LENGTH - rb->index;

    float *data_ptr = &rb->data[0];

    memcpy(out, (void *)(data_ptr + rb->index), sizeof(float) * first_partition);
    memcpy(out + first_partition, (void *)(data_ptr), sizeof(float) * rb->index);
}
