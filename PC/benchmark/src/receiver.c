/******************************************************************************
 * Title                 :   UDP-packer receiver
 * Filename              :   udp_receiver.c
 * Author                :   jteglund
 * Origin Date           :   20/06/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 9.5.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 Functions to manage a UDP-packet receiver with functionality to write to a circular buffer.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdbool.h>
#include "receiver.h"

/*
Create a ring buffer
*/
ring_buffer *create_ring_buffer()
{
    ring_buffer *rb = (ring_buffer *)calloc(1, sizeof(ring_buffer));
    rb->index = 0;
    return rb;
}

/*
Destroy a ring buffer
*/
ring_buffer *destroy_ring_buffer(ring_buffer *rb)
{
    free(rb);
    rb = NULL;
    return rb;
}

msg *create_msg()
{
    return (msg *)calloc(1, sizeof(msg));
}

msg *destroy_msg(msg *message)
{
    free(message);
    message = NULL;
    return message;
}

int create_and_bind_socket(bool replay_mode)
{
    int socket_desc;
    struct sockaddr_in server_addr;

    socket_desc = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

    if (socket_desc < 0)
    {
        printf("Error creating socket\n");
        return -1;
    }
    printf("Socket created successfully\n");

    // Set port and IP:
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(UDP_PORT);
    if (replay_mode)
    {
        server_addr.sin_addr.s_addr = inet_addr(UDP_REPLAY_IP);
    }
    else
    {
        server_addr.sin_addr.s_addr = inet_addr(UDP_IP);
    }

    // Bind to the set port and IP:
    if (bind(socket_desc, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        printf("Couldn't bind socket to the port\n");
        return -1;
    }
    printf("Binding complete\n");

    return socket_desc;
}

// int convertTo24Bit(int twoCom)
// {
//     if (twoCom & (1 << 8))
//     {
//         twoCom = ~twoCom;
//         int orig = twoCom >> 8;


//         orig += 1;

//         orig = -orig;
//         // printf("Orig\n");

//         return orig;
//     }

//     else {
//         return twoCom >> 8;
//     }
// }
#include <stdio.h>
void bin(unsigned n)
{
    unsigned i;
    for (i = 1 << 31; i > 0; i = i / 2)
        (n & i) ? printf("1") : printf("0");
}
int convertTo24Bit(int twoCom)
{
    // printf("\n");
    // bin((unsigned)twoCom);
    // printf("\n");
    // return twoCom;
    if (twoCom & (1 << 23))
    {


        int high = twoCom;
        high = ~high;
        // return high;
        int orig = -high;
        return orig;
        // // orig = (high << 24) | orig;



        // return orig >> 8;
    }

    else
    {
        return twoCom;
    }
}

int receive_and_write_to_buffer(int socket_desc, ring_buffer *rb, msg *message, int n_arrays)
{
    int msg_offset = 0;
    int step = 0;
    for (int i = 0; i < BUFFER_LENGTH; i += N_MICROPHONES)
    {
        
        // n_arrays = 3;
        // int si = 1458;
        // uint32_t msg1[1458];
        // recv(socket_desc, msg1, 1458, 0);
        if (recv(socket_desc, message, sizeof(msg), 0) < 0)
        // if (recv(socket_desc, message, si, 0) < 0)
        {
            printf("Couldn't receive\n");
            return -1;
        }

        // printf("%d %d %d %d\n", message->counter, message->frequency, message->n_arrays, message->protocol_ver);

        
        // printf("%d\n", sizeof(msg));
        // int orig = message->stream[3] << 8;
        // // int orig = msg1[3];
        // int num = convertTo24Bit(orig);
        // float val = ((float)num / 16777216);
        // bin(message->stream[3]);
        // printf("(%f, %d, %d) ", val, num, orig);
        // // printf("%d ", num);
        // // printf("%f ", val);

        // continue;

        // if (i%EVERY_N_SAMPLES != 0)
        // {
        //     continue;
        // }

        /*
        Fixes ordering such that the microphone data will be (microphone, data)

        with indexes as following:

        [0 1 2 3
         4 5 6 7
         8 9 ...]

        The array orders:
             ···
        15 14 13 12 11 10 9 8
         0  1  2  3  4  5 6 7

        */
        int s = 0;

        // for (int t = 0; t < 64; t++)
        // {
        //     rb->data[step + t * N_SAMPLES] = (float)((double)(message->stream[t]) / NORM_FACTOR);
        // }

#if 0
        for (int i = 0; i < N_MICROPHONES; i++)
        {
            rb->data[step + N_SAMPLES * i] = (float)((double)convertTo24Bit(message->stream[msg_offset + i] << 8) / NORM_FACTOR);
        }
        
        
#else
        for (int n = 0; n < n_arrays; n++)
        {
            for (int y = 0; y < ROWS; y++)
            {
                int row = n * ROWS * COLUMNS + y * COLUMNS;
                if ((y % 2) == 0)
                {
                    for (int x = 0; x < COLUMNS; x++)
                    {
                        // rb->data[step + N_SAMPLES * s] = (float)((double)(message->stream[msg_offset + row + x]) / NORM_FACTOR);
                        // rb->data[step + N_SAMPLES * s] = (float)((double)convertTo24Bit(message->stream[msg_offset + row + x]<<8) / NORM_FACTOR);
                        rb->data[step + N_SAMPLES * s] = (float)((double)convertTo24Bit(message->stream[msg_offset + row + x]) / NORM_FACTOR);

                        s++;
                    }
                }
                else
                {
                    for (int x = 0; x < COLUMNS; x++)
                    {
                        // rb->data[step + N_SAMPLES * s] = (float)((double)(message->stream[msg_offset + row + COLUMNS - x]) / NORM_FACTOR);
                        // rb->data[step + N_SAMPLES * s] = (float)((double)convertTo24Bit(message->stream[msg_offset + row + COLUMNS - x]<<8) / NORM_FACTOR);
                        rb->data[step + N_SAMPLES * s] = (float)((double)convertTo24Bit(message->stream[msg_offset + row + COLUMNS - x]) / NORM_FACTOR);
                        s++;
                    }
                }
            }
        }
#endif

        step++;
    }
    return 0;
}

int receive_header_data(int socket_desc)
{
    msg *client_msg = (msg *)calloc(1, sizeof(msg));
    if (recv(socket_desc, client_msg, sizeof(msg), 0) < 0)
    {
        printf("Couldn't receive\n");
        return -1;
    }
    int8_t n_arrays = client_msg->n_arrays;
    if (client_msg->protocol_ver != FPGA_PROTOCOL_VERSION)
    {
        return -1;
    }
    free(client_msg);
    return n_arrays;
}

int close_socket(int socket_desc)
{
    return close(socket_desc);
}