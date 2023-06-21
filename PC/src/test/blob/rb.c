#include <stdlib.h>
#include <stdio.h>
#include <time.h>      // time for seeding
#include <pthread.h>   // include pthread functions and structures
#include <semaphore.h> // include semaphores
#include <unistd.h>    // include sleep

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

#define BUFFER_SIZE 5

#define KEY 1234

typedef struct _rb
{
    float data[BUFFER_SIZE];
    int index;
} ring_buffer;

ring_buffer *rb;

int shmid; // Shared memory ID

int semid;                                   // Semaphore ID
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


void myread()
{
    semop(semid, &my_sem_wait, 1);

    float rout[BUFFER_SIZE];

    float *out = &rout[0];

    int first_partition = BUFFER_SIZE - rb->index;

    float *data_ptr = &rb->data[0];

    memcpy(out, (void *)(data_ptr + rb->index), sizeof(float) * first_partition);
    memcpy(out + first_partition, (void *)(data_ptr), sizeof(float) * rb->index);

    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        printf("%f ", out[i]);
    }

    printf("\n");

    semop(semid, &my_sem_signal, 1);
}

void mywrite(float *d, int length)
{
    semop(semid, &my_sem_wait, 1);

    printf("Writing\n");

    for (int i = 0; i < length; i++)
    {
        rb->data[rb->index] = d[i];

        rb->index = (rb->index + 1) % BUFFER_SIZE;
    }

    semop(semid, &my_sem_signal, 1);
}


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
    else if (pid == 0) // Child
    {
        float data[10] = {.1, .2, .3, .4, .5, .6, .7, .8, .9};
        for (int i = 0; i < 10; i++)
        {
            // data[0] = (float)i;
            mywrite(&data[0], 6);

            // mywrite((float)i, 1);
        }
        
    } else {
        // Parent
        for (int i = 0; i < 10; i++)
        {
            myread();
        }
        
    }

    return 0;
}
