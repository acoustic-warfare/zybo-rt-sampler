#include <stdio.h>
#include <stdlib.h>

// Semaphores
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>

#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "circular_buffer.h"
#include "udp_receiver.h"
#include "config.h"


#define KEY 1234

int shmid; // Shared memory ID

// struct ring_buffer* buffer;

struct ringba* rob;

// ring_buffer *rb = (ring_buffer *) calloc(1, sizeof(ring_buffer));

int semid; // Semaphore ID
struct sembuf sem_wait = {0, -1, SEM_UNDO}; // Wait operation
struct sembuf sem_signal = {0, 1, SEM_UNDO}; // Sig operation

/*
Create a shared memory ring buffer
*/
void init_shared_memory()
{
    // Create
    shmid = shmget(KEY, sizeof(struct ringba), IPC_CREAT);

    if (shmid == -1) {
        perror("shmget not working");
        exit(1);
    }

    rob = (struct ringba*)shmat(shmid, NULL, 0);

    if (rob == (struct ringba *)-1)
    {
        perror("shmat not working");
        exit(1);
    }

    rob->index = 0;

    // Init mem
    // rb->read_pos = 0;
    // rb->write_pos = 0;
}

/*
Create interprocess semaphore
*/
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

// void test_rb()
// {
//     // Initialize the ring buffer
//     ring_buffer *myRingBuffer = (ring_buffer *)calloc(1, sizeof(ring_buffer));
//     myRingBuffer->index = 0;
// 
//     // float d[10] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
// 
//     float d[BUFFER_LENGTH * 2];
// 
//     for (int i = 0; i < BUFFER_LENGTH * 2; i++)
//     {
//         d[i] = (float)(i + 1);
//     }
// 
//     float *ptr = d;
// 
//     write_buffer(myRingBuffer, ptr, BUFFER_LENGTH, 0);
// 
//     write_buffer(myRingBuffer, ptr, BUFFER_LENGTH - 4, 0);
// 
//     float out[BUFFER_LENGTH] = {0.0};
// 
//     for (size_t i = 0; i < BUFFER_LENGTH; i++)
//     {
//         out[i] = 0.0;
//     }
// 
//     // convolve_avx_unrolled_vector_unaligned_fma()
// 
//     read_buffer_mcpy(myRingBuffer, &out[0]);
// 
//     for (int i = 0; i < BUFFER_LENGTH; i++)
//     {
//         printf("%f ", out[i]);
//     }
// 
//     printf("\n");
// }

//void get_data(float *out) {
//    if (semop(semid, &sem_wait, 1) == -1)
//    {
//        perror("semop");
//        exit(1);
//    }
//
//    read_mcpy(rob, out);
//
//    if (semop(semid, &sem_signal, 1) == -1)
//    {
//        perror("semop");
//        exit(1);
//    }
//}

int main(int argc, char const *argv[])
{
    
    init_shared_memory();

    init_semaphore();

    

    pid_t pid = fork(); // Fork child

    if (pid == -1)
    {
        // printf("lonk\n");
        perror("fork");
        exit(1);
    }
    else if (pid == 0)
    { // Child

        // Create UDP socket:
        int socket_desc = create_and_bind_socket();

        while (1)
        {
            if (semop(semid, &sem_wait, 1) == -1)
            {
                perror("semop");
                exit(1);
            }

            receive_and_write_to_buffer_test(socket_desc, rob);
            printf("\nWriting\n");
            // Signal unlock (DONE)
            if (semop(semid, &sem_signal, 1) == -1)
            {
                perror("semop");
                exit(1);
            }
        }

        close_socket(socket_desc);
    }
    else
    {
        float out[BUFFER_LENGTH] = {0.0};

        for (size_t i = 0; i < BUFFER_LENGTH; i++)
        {
            out[i] = 0.0;
        }

        while (1)
        {
            if (semop(semid, &sem_wait, 1) == -1)
            {
                perror("semop");
                exit(1);
            }

            read_mcpy(rob, &out[0]);

            if (semop(semid, &sem_signal, 1) == -1)
            {
                perror("semop");
                exit(1);
            }

            for (int i = 0; i < 20; i++)
            {
                printf("%f ", out[i]);
            }

            printf("Reading\n");

            sleep(1);
        }
    }

    

    // test_rb();
    return 0;
}
