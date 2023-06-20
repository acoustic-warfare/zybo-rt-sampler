#include "circular_buffer.h"

#define UDP_PORT 21844
#define SERVER_IP "10.0.0.1"

/// @brief FPGA Protocol
typedef struct _msg
{
    int32_t array_id;
    int32_t version;
    int32_t frequency;
    int32_t counter;
    int32_t stream[64]; // Change magic number
} msg;

/// @brief Creates and binds the socket to a server ip and port.
/// @pre Requires the SERVER_IP and UDP_PORT to be correctly specified in the header file.
/// @return A socket descriptor if successfully created and bound. -1 if an error occured.
int create_and_bind_socket();

/// @brief Receives messages from the UDP client forever and prints the messages.
/// @param socket_desc A socket file descriptor
/// @return -1 if error occured.
int receive_and_print(int socket_desc);

// TODO
int receive_and_write_to_buffer(int socket_desc, ring_buffer *rb);

/// @brief Closes the socket descriptor.
/// @param socket_desc A socket file descriptor.
/// @return -1 if error occured.
int close_socket(int socket_desc);