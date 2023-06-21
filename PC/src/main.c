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

#define BUFFER_LENGTH 10

// typedef struct _rb
// {
//     float data[BUFFER_LENGTH];
//     int index;
// } ring_buffer;

ring_buffer *rb;

int shmid; // Shared memory ID

int semid;                                      // Semaphore ID
struct sembuf my_sem_wait = {0, -1, SEM_UNDO};  // Wait operation
struct sembuf my_sem_signal = {0, 1, SEM_UNDO}; // Sig operation

/*
Create a shared memory ring buffer
*/
void init_shared_memory()
{
    // Create
    shmid = shmget(KEY, sizeof(ring_buffer), IPC_CREAT);

    if (shmid == -1)
    {
        perror("shmget not working");
        exit(1);
    }

    rb = (ring_buffer *)shmat(shmid, NULL, 0);

    if (rb == (ring_buffer *)-1)
    {
        perror("shmat not working");
        exit(1);
    }

    rb->index = 0;
}

void init_semaphore()
{
    semid = semget(KEY, 1, IPC_CREAT);

    if (semid == -1)
    {
        perror("semget");
        exit(1);
    }

    union semun
    {
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

void _myread()
{
    semop(semid, &my_sem_wait, 1);

    float rout[BUFFER_LENGTH];

    float *out = &rout[0];

    int first_partition = BUFFER_LENGTH - rb->index;

    float *data_ptr = &rb->data[0];

    memcpy(out, (void *)(data_ptr + rb->index), sizeof(float) * first_partition);
    memcpy(out + first_partition, (void *)(data_ptr), sizeof(float) * rb->index);

    printf("\n");
    for (int i = 0; i < BUFFER_LENGTH; i++)
    {
        printf("%f ", rout[i]);
    }

    printf("\n");

    semop(semid, &my_sem_signal, 1);
}

void myread()
{
    semop(semid, &my_sem_wait, 1);
    printf("\n");
    for (int i = 0; i < BUFFER_LENGTH; i++)
    {
        printf("%f ", rb->data[i]);
    }

    printf("\n");

    semop(semid, &my_sem_signal, 1);
}

int main(int argc, char const *argv[])
{
    init_shared_memory();

    for (int i = 0; i < BUFFER_LENGTH; i++) {
        rb->data[i] = 0.0;
    }

    init_semaphore();

    pid_t pid = fork(); // Fork child

    if (pid == -1)
    {
        // printf("lonk\n");
        perror("fork");
        exit(1);
    }
    else if (pid == 0) // Child
    {
        // Create UDP socket:
        int socket_desc = create_and_bind_socket();

        msg *client_msg = (msg *)calloc(1, sizeof(msg));

        //int counter = 0;

        while (1)
        {
            if (recv(socket_desc, client_msg, sizeof(msg), 0) < 0)
            {
                printf("Couldn't receive\n");
                return -1;
            }
            //client_msg->stream[0] = client_msg->counter;

            semop(semid, &my_sem_wait, 1);

            //for (int i = 0; i < 1; i++)
            //{
            //    rb->data[rb->index] = (float)client_msg->stream[i];
// //
            //    rb->index = (rb->index + 1) % BUFFER_LENGTH;
            //}

            rb->data[rb->index] = (float)client_msg->counter;

            int buffer_length = BUFFER_LENGTH - 1;
            int previous_item = rb->index;

            int idx = (previous_item) & buffer_length; // Wrap around

            // rb->data[idx] = (double)client_msg->counter;

            // rb->data[idx] = (float)client_msg->counter;

            

            printf("WRITER %f %f %d\n", rb->data[idx], (float)client_msg->counter, rb->index);

            // Sync current index
            // rb->index += 1;
            // rb->index &= BUFFER_LENGTH - 1;
            //for (int i = 0; i < BUFFER_LENGTH; i++)
            //{
            //    printf("%f ", rb->data[i]);
            //}
//
            //printf("\n");
            //
            // //rb->data[rb->index] = (float)counter;

            // printf("%f %d %d\n", rb->data[rb->index], rb->index, client_msg->counter);
            // 
            rb->index = (rb->index + 1) % BUFFER_LENGTH;

            // printf("\n%d\n", rb->index);

            semop(semid, &my_sem_signal, 1);
        }

        free(client_msg);

        close_socket(socket_desc);
    }
    else
    {
        while (1)
        {
            myread();
            // sleep(1);
            //printf("%d\n", rb->index);
        }
    }

    return 0;
}