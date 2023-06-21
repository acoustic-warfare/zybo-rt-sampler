#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define BUFFER_SIZE 10

struct ring_buffer
{
    char data[BUFFER_SIZE];
    int read_pos;
    int write_pos;
};

struct ring_buffer *buffer;


void writer(const char* data, int length) 
{

}

void reader(char *dest, int length)
{

}



int main(int argc, char const *argv[])
{
    pid_t pid = fork(); // Fork child

    if (pid == -1)
    {
        printf("lonk\n");
        perror("fork");
        exit(1);
    }
    else if (pid == 0)
    { // Child
        char data[BUFFER_SIZE];

        printf("Bonk\n");

        while (1)
        {
            printf("Lol\n");
            reader(&data[0], BUFFER_SIZE);

            printf("Data: %s\n", data);

            sleep(1);
        }
    }
    else
    {
        printf("Donk\n");
        // Parent process
        const char *message = "Hello World!";

        int message_length = strlen(message);

        while (1)
        {
            printf("sBonk");
            printf("Donk\n");
            writer(message, message_length);

            sleep(1);
        }
    }
    return 0;
}
