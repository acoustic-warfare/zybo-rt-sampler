#define UDP_PORT 21844
#define SERVER_IP "10.0.0.1"


/// @brief Creates and binds the socket to a server ip and port.
/// @pre Requires the SERVER_IP and UDP_PORT to be correctly specified in the header file.
/// @return A socket descriptor if successfully created and bound. -1 if an error occured.
int create_and_bind_socket();

/// @brief Receives messages from the UDP client forever and prints the messages.
/// @param socket_desc A socket file descriptor
/// @return -1 if error occured.
int receive_and_print_message(int socket_desc);

// TODO
int receive_and_write_to_buffer(int socket_desc);

/// @brief Closes the socket descriptor.
/// @param socket_desc A socket file descriptor.
/// @return -1 if error occured.
int close_socket(int socket_desc);