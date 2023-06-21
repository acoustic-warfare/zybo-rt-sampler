#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define BUFFER_SIZE 10

#define KEY 1234

struct ring_buffer {
    int data[BUFFER_SIZE];
    int read_pos;
    int write_pos;
};



int shmid; // Shared memory ID

struct ring_buffer* buffer;

int semid; // Semaphore ID
struct sembuf sem_wait = {0, -1, SEM_UNDO}; // Wait operation
struct sembuf sem_signal = {0, 1, SEM_UNDO}; // Sig operation

void init_shared_memory()
{
    // Create
    shmid = shmget(KEY, sizeof(struct ring_buffer), IPC_CREAT);

    if (shmid == -1) {
        perror("shmget not working");
        exit(1);
    }

    buffer = (struct ring_buffer*)shmat(shmid, NULL, 0);

    if (buffer == (struct ring_buffer*)-1) {
        perror("shmat not working");
        exit(1);
    }

    // Init mem
    buffer->read_pos = 0;
    buffer->write_pos = 0;
}

void init_semaphore()
{
    semid = semget(KEY, 1, IPC_CREAT);

    if (semid == -1) {
        perror("semget");
        exit(1);
    }

    union semun {
        int val;
        struct semid_ds *buf;
        short *array;
    } argument;
    argument.val = 1;

    // Set semaphore to 1
    if (semctl(semid, 0, SETVAL, argument) == -1)
    {
        perror("semctl");
        exit(1);
    }
}


void writer(const int* data, int length)
{
    if (semop(semid, &sem_wait, 1) == -1)
    {
        perror("semop");
        exit(1);
    }

    // Write to buffer
    for (int i = 0; i < length; i++)
    {
        buffer->data[buffer->write_pos] = data[i];
        buffer->write_pos = (buffer->write_pos + 1) % BUFFER_SIZE;
    }

    // Signal unlock (DONE)
    if (semop(semid, &sem_signal, 1) == -1) {
        perror("semop");
        exit(1);
    }
}


void reader(int *dest, int length)
{
    if (semop(semid, &sem_wait, 1) == -1)
    {
        perror("semop");
        exit(1);
    }


    for (int i = 0; i < length; i++)
    {
        dest[i] = buffer->data[buffer->read_pos];
        buffer->read_pos = (buffer->read_pos + 1) % BUFFER_SIZE;
    }

    // Signal unlock (DONE)
    if (semop(semid, &sem_signal, 1) == -1)
    {
        perror("semop");
        exit(1);
    }
}

int main(int argc, char const *argv[])
{
    // printf("onk\n");
    init_shared_memory();
    // printf("onk\n");
    init_semaphore();

    pid_t pid = fork(); // Fork child

    if (pid == -1)
    {
        // printf("lonk\n");
        perror("fork");
        exit(1);
    } else if (pid == 0) { // Child
        int data[BUFFER_SIZE] = {};

        // printf("Bonk\n");

        while (1) {
            // printf("Lol\n");
            reader(&data[0], BUFFER_SIZE);

            printf("Data:\n");

            for (int i = 0; i < BUFFER_SIZE; i++)
            {
                printf("%d ", data[i]);
            }
            

            // printf("Data: %d\n", data);

            usleep(1000);
        }
    } else {
        // printf("Donk\n");
        // Parent process
        const char* message = ".";

        int message_length = strlen(message);

        int i[5] = {1, 2, 3, 4, 5};

        int *dat = &i[0];

        

        while (1) {
            // printf("sBonk");
            // printf("Donk\n");
            // writer(message, message_length);
            writer(dat, 5);

            sleep(1);
        }
    }


    return 0;
}
