#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

// Note power of two buffer size
#define BUFFER_LENGTH 256

typedef struct _ring_buffer
{
    int index;
    int size;
    float data[BUFFER_LENGTH];
} ring_buffer;


/**
Write data from an address `in` to a ring buffer you can specify offset 
but most of the times, it will probably just be 0
*/
void write_buffer(ring_buffer *rb, float *in, int length, int offset)
{
    // -1 for our binary modulo in a moment
    int buffer_length = rb->size - 1;
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
    rb->index &= rb->size - 1;
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
    for (size_t i = rb->index; i < rb->size; i++)
    {
        if (rb->size - length <= index)
        {
            out[index + offset - (rb->size - length)] = rb->data[i];
        }
        
        index++;
    }

    for (size_t i = 0; i < rb->index; i++)
    {
        if (rb->size - length <= index)
        {
            out[index + offset - (rb->size - length)] = rb->data[i];
        }
        
        index++;
    }
}

void read_buffer_mcpy(ring_buffer *rb, float *out, int length, int offset)
{
    int first_partition = rb->size - rb->index;

    float *data_ptr = &rb->data[0];

    memcpy(out, (void *)(data_ptr + rb->index), sizeof(float) * first_partition);
    memcpy(out + first_partition, (void *)(data_ptr), sizeof(float) * rb->index);
}

int main(int argc, char const *argv[])
{
    // Initialize the ring buffer
    ring_buffer *myRingBuffer = (ring_buffer *)calloc(1, sizeof(ring_buffer));
    myRingBuffer->size = BUFFER_LENGTH;
    myRingBuffer->index = 0;

    //float d[10] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    float d[BUFFER_LENGTH * 2];

    for (size_t i = 0; i < BUFFER_LENGTH * 2; i++)
    {
        d[i] = (float)(i+1);
    }
    

    float *ptr = d;

    //for (size_t i = 0; i < 6; i++)
    //{
    //    writeIntoBuffer(myRingBuffer, ptr + i, 1);
    //}

    //writeIntoBuffer(myRingBuffer, ptr, 3);
    //writeIntoBuffer(myRingBuffer, ptr, 3);

    write_buffer(myRingBuffer, ptr, 300, 0);

    float out[BUFFER_LENGTH] = {0.0};

    //retrieve_scalar(&out[0], myRingBuffer);

    //read_buffer(myRingBuffer, &out[0], 10, 0);

    read_buffer_mcpy(myRingBuffer, &out[0], 10, 0);

        for (size_t i = 0; i < BUFFER_LENGTH; i++)
    {
        printf("%f ", out[i]);
    }
    

    

    //for (size_t i = myRingBuffer->currentIndex; i < myRingBuffer->sizeOfBuffer; i++)
    //{
    //    printf("%f ", myRingBuffer->data[i]);
    //}
//
    //for (size_t i = 0; i < myRingBuffer->currentIndex; i++)
    //{
    //    printf("%f ", myRingBuffer->data[i]);
    //}
    

    //for (size_t i = 0; i < 6; i++)
    //{
    //    printf("%f ", myRingBuffer->data[i]);
    //}

    printf("\n");
    
    

    

    return 0;
}
