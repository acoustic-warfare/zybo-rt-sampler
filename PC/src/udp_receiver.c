#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdbool.h>

#define HEADER 4
#define N_MICROPHONES 2
#define N_SAMPLES 32
#define BUFFER_LENGTH HEADER + N_MICROPHONES * N_SAMPLES

#define UDP_PORT 21844
#define SERVER_IP "10.0.0.1"

int main(void){
    int socket_desc;
    struct sockaddr_in server_addr, client_addr;
    int32_t client_message[BUFFER_LENGTH];
    int client_struct_length = sizeof(client_addr);
    
    // Clean buffers:
    memset(client_message, '\0', sizeof(client_message));
    
    // Create UDP socket:
    socket_desc = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    
    if(socket_desc < 0){
        printf("Error while creating socket\n");
        return -1;
    }
    printf("Socket created successfully\n");
    
    // Set port and IP:
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(UDP_PORT);
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    
    // Bind to the set port and IP:
    if(bind(socket_desc, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0){
        printf("Couldn't bind to the port\n");
        return -1;
    }
    printf("Done with binding\n");
    
    printf("Listening for incoming messages...\n\n");
    
    // Receive client's message:
    while(true){
        if (recvfrom(socket_desc, client_message, sizeof(client_message), 0,
            (struct sockaddr*)&client_addr, &client_struct_length) < 0){
            printf("Couldn't receive\n");
            return -1;
        }
        printf("%s" "%d \n", "Sample count: ", client_message[3]);
        
    }

    
    // Close the socket:
    close(socket_desc);
    
    return 0;
}