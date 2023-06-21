
#include <sys/socket.h>
#include <arpa/inet.h> //inet_addr
#include <unistd.h>    //write

#include <unistd.h>    //write
#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PORT 9999
#define ADDRESS "10.0.0.1"

#define N_SAMPLES 64

#pragma pack(1)

// Version 1.0 (BatMobile 1000)
typedef struct payload_protocol_t
{
    int id;               // Transfer id, for tracking later
    int protocol_version; // To trace error
    int fs;               // Sampling rate
    int fs_nr;            // Sample number
    int samples;          // Every mic
    int sample_error;     // If error inside
    int bitstream[N_SAMPLES];   // The bitstream from the mic array
} payload_protocol;

#pragma pack()

// Sleep function for ms
int msleep(unsigned int tms)
{
    return usleep(tms * 1000);
}

void transmitMicArraydata(int sock, void *ctx, uint32_t ctxsize)
{
    // Print error message when writing to socket fails
    if (write(sock, ctx, ctxsize) < 0)
    {
        printf("Error sending message.");
        close(sock);
        exit(1);
    }
    return;
}

int main(int argc, char const *argv[])
{
    const int port = PORT;
    const char * address = ADDRESS;

    int sock;

    time_t t;

    srand((unsigned)time(&t));

    return 0;
}
