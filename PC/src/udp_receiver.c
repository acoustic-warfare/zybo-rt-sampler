#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdbool.h>
#include "udp_receiver.h"

//TODO move these defines to config
#define HEADER 4
#define N_MICROPHONES 2
#define N_SAMPLES 32
#define BUFFER_LENGTH HEADER + N_MICROPHONES * N_SAMPLES

int create_and_bind_socket(){
    int socket_desc;
    struct sockaddr_in server_addr;

    socket_desc = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    
    if(socket_desc < 0){
        printf("Error creating socket\n");
        return -1;
    }
    printf("Socket created successfully\n");
    
    // Set port and IP:
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(UDP_PORT);
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    
    // Bind to the set port and IP:
    if(bind(socket_desc, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0){
        printf("Couldn't bind socket to the port\n");
        return -1;
    }
    printf("Binding complete\n");

    return socket_desc;
}

int receive_and_print_message(int socket_desc){
    // Create buffer
    int32_t client_message[BUFFER_LENGTH];
    // Clean buffers:
    memset(client_message, '\0', sizeof(client_message));

    printf("Listening for incoming messages...\n\n");
    
    // Receive client's message:
    while(true){
        if (recv(socket_desc, client_message, sizeof(client_message), 0) < 0){
            printf("Couldn't receive\n");
            return -1;
        }
        //Prints the 3rd element of the message received
        printf("%s" "%d \n", "Sample count: ", client_message[3]);
    }

    return 0;
}

int receive_and_write_to_buffer(int socket_desc){
    return 0;
}

int close_socket(int socket_desc){
    return close(socket_desc);
}

int main(void){
    // Create UDP socket:
    int socket_desc = create_and_bind_socket();
    
    receive_and_print_message(socket_desc);
    // Close the socket:
    close_socket(socket_desc);
    
    return 0;
}
