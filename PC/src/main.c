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
#include "config.h"

ring_buffer *rb;
msg *client_msg;
int shmid; // Shared memory ID
int semid;                                      // Semaphore ID
int socket_desc;

struct sembuf my_sem_wait = {0, -1, SEM_UNDO};  // Wait operation
struct sembuf my_sem_signal = {0, 1, SEM_UNDO}; // Sig operation

void signal_handler(){
    //Remove shared memory and semafores
    shmctl(shmid, IPC_RMID, NULL);
    semctl(semid, 0, IPC_RMID);
    close_socket(socket_desc);
    destroy_msg(client_msg);
    exit(-1);
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


void myread(float *out)
{
    semop(semid, &my_sem_wait, 1);
    memcpy(out, (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &my_sem_signal, 1);
}

int load()
{
    signal(SIGINT, signal_handler);
    signal(SIGKILL, signal_handler);
    signal(SIGTERM, signal_handler);
    
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
        socket_desc = create_and_bind_socket();
        client_msg = create_msg();

        while (1)
        {
            semop(semid, &my_sem_wait, 1);

            if(receive_and_write_to_buffer(socket_desc, rb, client_msg) == -1){
                return -1;
            }

            semop(semid, &my_sem_signal, 1);
            continue;
        }
    }

    return 0;
}

int main(int argc, char const *argv[])
{
    load();
    float out[BUFFER_LENGTH] = {0.0};
    while(1){
        myread(&out[0]);
        printf("%f", out[0]);
    }
    
    return 0;
}