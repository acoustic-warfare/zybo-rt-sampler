/******************************************************************************
 * Title                 :   UDP-packer receiver
 * Filename              :   udp_receiver.h
 * Author                :   jteglund
 * Origin Date           :   20/06/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 9.5.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 Functions to manage a UDP-packet receiver with functionality to write to a circular buffer.

 USAGE:

 #include "udp_receiver.h"

 sockfd = create_and_bind_socket();
 receive_and_print(sockfd); | receive_and_write_to_buffer(sockfd, ring_buffer *rb);
 close_socket();
*/

#ifndef _RECEIVER_H_
#define _RECEIVER_H_

#include "config.h"
#include <stdbool.h>

#define BUFFER_SIZE BUFFER_LENGTH * 4

typedef struct _ring_buffer
{
    int index;
    float data[BUFFER_SIZE];
    int counter;
} ring_buffer;

// typedef struct _ring_buffer
// {
//     int index;
//     float data[BUFFER_LENGTH];
//     // double mydata[BUFFER_LENGTH];
//     int counter;
// } ring_buffer;

ring_buffer *create_ring_buffer();

ring_buffer *destroy_ring_buffer(ring_buffer *rb);

/// @brief FPGA Protocol Version 2
typedef struct _msg
{
    u_int16_t frequency;
    int8_t n_arrays;
    int8_t protocol_ver;
    int32_t counter;
    
    int32_t stream[N_MICROPHONES];
} msg;

msg *create_msg();
msg *destroy_msg(msg *msg);

/// @brief Creates and binds the socket to a server ip and port.
/// @pre Requires the SERVER_IP and UDP_PORT to be correctly specified in the header file.
/// @return A socket descriptor if successfully created and bound. -1 if an error occured.
int create_and_bind_socket(bool replay_mode);

/// @brief Receives messages from UDP client forever and writes to a ring buffer with the FPGA Protocol version 2
/// @param socket_desc A socket file descriptor
/// @param rb A pointer to a ring buffer
/// @param message A pointer for temporarily storing received message
/// @param n_arrays Number of connected arrays
/// @return 0 if no errors and -1 if the message can't be received
int receive_and_write_to_buffer(int socket_desc, ring_buffer *rb, msg *message, int n_arrays);

/// @brief Closes the socket descriptor.
/// @param socket_desc A socket file descriptor.
/// @return -1 if error occured.
int close_socket(int socket_desc);

/// @brief Receives the first message and returns the number of arrays
/// @param socket_desc
/// @return The number of connected arrays
int receive_header_data(int socket_desc);

void receive_to_buffer(int socket_desc, float *out, msg *message, int n_arrays);

#endif