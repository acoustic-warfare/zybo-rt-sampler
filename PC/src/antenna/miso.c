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
#include <sys/socket.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

#include "circular_buffer.h"
#include "udp_receiver.h"
#include "delay.h"
#include "miso.h"
#include "antenna.h"
#include "config.h"

miso_ipc *mi;

ring_buffer *rb;
msg *client_msg;

int shmid_ringbuffer; // Shared memory ID
int semid_ringbuffer;                                      // Semaphore ID

int shmid_miso; // Shared memory ID
int semid_miso; 

int socket_desc;

struct sembuf ringbuffer_sem_wait = {0, -1, SEM_UNDO};  // Wait operation
struct sembuf ringbuffer_sem_signal = {0, 1, SEM_UNDO}; // Sig operation

struct sembuf miso_sem_wait = {0, -1, SEM_UNDO};  // Wait operation
struct sembuf miso_sem_signal = {0, 1, SEM_UNDO}; // Sig operation

void signal_handler(){
    //Remove shared memory and semafores
    shmctl(shmid_ringbuffer, IPC_RMID, NULL); // Delete rb
    semctl(semid_ringbuffer, 0, IPC_RMID);

    shmctl(shmid_miso, IPC_RMID, NULL); // Delete rb
    semctl(semid_miso, 0, IPC_RMID);

    close_socket(socket_desc);
    destroy_msg(client_msg);
    exit(-1);
}

/*
Create a shared memory ring buffer
*/
void init_shared_memory_ringbuffer()
{
    // Create
    shmid_ringbuffer = shmget(KEY, sizeof(ring_buffer), IPC_CREAT | 0666);

    if (shmid_ringbuffer == -1)
    {
        perror("shmget not working");
        exit(1);
    }

    rb = (ring_buffer *)shmat(shmid_ringbuffer, NULL, 0);

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

void init_shared_memory_miso()
{
    shmid_miso = shmget(KEY, sizeof(miso_ipc), IPC_CREAT | 0666);

    if (shmid_miso == -1)
    {
        perror("shmget not working");
        exit(1);
    }

    mi = (miso_ipc *)shmat(shmid_miso, NULL, 0);

    if (mi == (miso_ipc *)-1)
    {
        perror("shmat not working");
        exit(1);
    }

    // Setup initial data
    mi->x = 0.0;
    mi->y = 0.0;

    for (int i = 0; i < N_SAMPLES; i++)
    {
        mi->data[i] = 0.0;
    }
}

void init_semaphore_ringbuffer()
{
    semid_ringbuffer = semget(KEY, 1, IPC_CREAT | 0666);

    if (semid_ringbuffer == -1)
    {
        perror("semget");
        exit(1);
    }

    union semun
    {
        int val;
        struct semid_ringbuffer_ds *buf;
        short *array;
    } argument;
    argument.val = 1;

    // Set semaphore to 1
    if (semctl(semid_ringbuffer, 0, SETVAL, argument) == -1)
    {
        perror("semctl");
        exit(1);
    }
}

void init_semaphore_miso()
{
    semid_miso = semget(KEY, 1, IPC_CREAT | 0666);

    if (semid_miso == -1)
    {
        perror("semget");
        exit(1);
    }

    union semun
    {
        int val;
        struct semid_miso_ds *buf;
        short *array;
    } argument;
    argument.val = 1;

    // Set semaphore to 1
    if (semctl(semid_miso, 0, SETVAL, argument) == -1)
    {
        perror("semctl");
        exit(1);
    }
}

int receive_loop()
{
    // Create UDP socket:
    socket_desc = create_and_bind_socket();
    client_msg = create_msg();

    while (1)
    {
        semop(semid_ringbuffer, &ringbuffer_sem_wait, 1);

        // TODO might not release semaphore
        if(receive_and_write_to_buffer(socket_desc, rb, client_msg) == -1){
            return -1;
        }

        semop(semid_ringbuffer, &ringbuffer_sem_signal, 1);
    }
}


int steer(float new_x, float new_y)
{
    semop(semid_miso, &miso_sem_wait, 1);
    mi->x = new_x;
    mi->y = new_y;
    semop(semid_miso, &miso_sem_signal, 1);
    return 0;
}

void miso_listen(float *out)
{
    semop(semid_miso, &miso_sem_wait, 1);
    memcpy(out, (void *)&mi->data[0], sizeof(float) * N_SAMPLES);
    semop(semid_miso, &miso_sem_signal, 1);
}

void beamformer()
{

    float **tmp_coefficients = (float **)malloc(N_MICROPHONES * sizeof(float *));

    for (int i = 0; i < N_MICROPHONES; i++)
    {
        tmp_coefficients[i] = (float *)malloc(N_TAPS * sizeof(float));
    }

    float mic_data[BUFFER_LENGTH];
    
    while (1)
    {
        // Retrieve latest data
        semop(semid_ringbuffer, &ringbuffer_sem_wait, 1);
        memcpy(mic_data, (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
        semop(semid_ringbuffer, &ringbuffer_sem_signal, 1);

        semop(semid_miso, &miso_sem_wait, 1);
        //directional_antenna_delay_coefficients((double)mi->x, (double)mi->y, 8, 8, 0.02, 48828.0, 340.0, tmp_coefficients);
        

        // Zero out entire array
        //memset((void *)&mi->data[0], 0, (N_SAMPLES) * sizeof(float));

        for (int i = 0; i < N_SAMPLES; i++)
        {
            mi->data[i] = mic_data[i];
        }
        
        //for (int i = 0; i < N_MICROPHONES; i++)
        //{
        //    delay_vectorized_add(&mic_data[i * N_SAMPLES], tmp_coefficients[i], (void *)&mi->data[0]);
        //}

        semop(semid_miso, &miso_sem_signal, 1);
    }
}

int load()
{

    signal(SIGINT, signal_handler);
    signal(SIGKILL, signal_handler);
    signal(SIGTERM, signal_handler);

    init_shared_memory_ringbuffer();
    init_shared_memory_miso();

    init_semaphore_ringbuffer();
    init_semaphore_miso();

    pid_t c1_pid, c2_pid;

    (c1_pid = fork()) && (c2_pid = fork()); // Creates two children

    if (c1_pid == 0) {
        /* Child 1 code goes here */
        receive_loop();
    } else if (c2_pid == 0) {
        /* Child 2 code goes here */
        beamformer();
    } else {
        return 0; /* Parent code goes here */
    }
}

