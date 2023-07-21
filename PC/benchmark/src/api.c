// Semaphores
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/types.h>

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include "config.h"
#include "receiver.h"

ring_buffer *rb; // Data to be stored in

msg *client_msg;

int shmid; // Shared memory ID
int semid; // Semaphore ID

int socket_desc;

struct sembuf data_sem_wait = {0, -1, SEM_UNDO};  // Wait operation
struct sembuf data_sem_signal = {0, 1, SEM_UNDO}; // Sig operation

pid_t pid_child;

/*
Remove shared memory and semafores
*/
void signal_handler()
{
    shmctl(shmid, IPC_RMID, NULL);
    semctl(semid, 0, IPC_RMID);
    close_socket(socket_desc);
    destroy_msg(client_msg);
}

/*
Stops the receiving child process
*/
void stop_receiving()
{
    signal_handler();
    kill(pid_child, SIGKILL);
}

/*
Create a shared memory ring buffer
*/
void init_shared_memory()
{
    // Create
    shmid = shmget(KEY, sizeof(ring_buffer), IPC_CREAT | 0666);

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
    rb->counter = 0;
    for (int i = 0; i < BUFFER_LENGTH; i++)
    {
        rb->data[i] = 0.0;
    }
}

/*
Create the semaphore
*/
void init_semaphore()
{
    semid = semget(KEY, 1, IPC_CREAT | 0666);

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

/*
Retrieve the data located in the ring buffer
*/
void get_data(float *out)
{
    semop(semid, &data_sem_wait, 1);
    memcpy(out, (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &data_sem_signal, 1);
}


/*
Main initialization function which starts a child process that continiously receive data
*/
int load(bool replay_mode)
{

    init_shared_memory();
    init_semaphore();

    pid_t pid = fork(); // Fork child

    if (pid == -1)
    {
        perror("fork");
        exit(1);
    }
    else if (pid == 0) // Child
    {
        // Create UDP socket:
        socket_desc = create_and_bind_socket(replay_mode);
        if (socket_desc == -1)
        {
            return -1;
        }
        client_msg = create_msg();
        int n_arrays = receive_header_data(socket_desc);
        if (n_arrays == -1)
        {
            return -1;
        }
        while (1)
        {
            semop(semid, &data_sem_wait, 1);

            if (receive_and_write_to_buffer(socket_desc, rb, client_msg, n_arrays) == -1)
            {
                return -1;
            }

            semop(semid, &data_sem_signal, 1);
        }
    }

    pid_child = pid;

    // Return to parent
    return 0;
}



// Algorithms

#include "algorithms/pad_and_sum.h"
void pad_mimo(float *image, int *adaptive_array, int n)
{
    float signals[BUFFER_LENGTH];

    get_data(&signals[0]);

    mimo_pad(&signals[0], image, adaptive_array, n);
}


#include "algorithms/convolve_and_sum.h"
void convolve_mimo_vectorized(float *image, int *adaptive_array, int n)
{
    float signals[BUFFER_LENGTH];

    get_data(&signals[0]);

    mimo_convolve_vectorized(&signals[0], image, adaptive_array, n);
}

void convolve_mimo_naive(float *image, int *adaptive_array, int n)
{
    float signals[BUFFER_LENGTH];

    get_data(&signals[0]);

    mimo_convolve_naive(&signals[0], image, adaptive_array, n);
}