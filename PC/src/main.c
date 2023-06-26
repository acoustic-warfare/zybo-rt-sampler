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

#define KEY 1234

#define THRESHOLD 256

// #define BUFFER_LENGTH 10

// typedef struct _rb
// {
//     float data[BUFFER_LENGTH];
//     int index;
// } ring_buffer;

ring_buffer *rb;

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

void __myread()
{
    
    
/*
    int length = BUFFER_LENGTH;
    int offset = 0;
*/
    float out[BUFFER_LENGTH];

//    int index = 0;
    semop(semid, &my_sem_wait, 1);

    int first_partition = BUFFER_LENGTH - rb->index;

    float *data_ptr = &rb->data[0];

    memcpy(out, (void *)(data_ptr + rb->index), sizeof(float) * first_partition);
    memcpy(out + first_partition, (void *)(data_ptr), sizeof(float) * rb->index);

    semop(semid, &my_sem_signal, 1);

    printf("\n");
    for (int i = 0; i < BUFFER_LENGTH; i++)
    {
        printf("%f ", out[i]);
    }

    printf("\n");
}

void myread(float *out)
{

    semop(semid, &my_sem_wait, 1);

    //read_buffer_mcpy(rb, out);
    memcpy(out, (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);

    semop(semid, &my_sem_signal, 1);

    //while (1)
    //{
    //    semop(semid, &my_sem_wait, 1);
//
    //    if (rb->counter > THRESHOLD - 2)
    //    {
    //        read_buffer_mcpy(rb, out);
    //        rb->counter = 0;
    //        semop(semid, &my_sem_signal, 1);
    //        break;
//
    //    }
//
    //    semop(semid, &my_sem_signal, 1);
    //}
    
}

int load()
{
    signal(SIGINT, signal_handler);
    signal(SIGKILL, signal_handler);
    signal(SIGTERM, signal_handler);
    
    init_shared_memory();

    for (int i = 0; i < BUFFER_LENGTH; i++)
    {
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
        socket_desc = create_and_bind_socket();

        msg *client_msg = (msg *)calloc(1, sizeof(msg));

        // int counter = 0;

        while (1)
        {

            semop(semid, &my_sem_wait, 1);

            for (int i = 0; i < BUFFER_LENGTH; i+=N_MICROPHONES)
            {
                if (recv(socket_desc, client_msg, sizeof(msg), 0) < 0)
                {
                    printf("Couldn't receive\n");
                    return -1;
                }

                for (int k = 0; k < N_MICROPHONES; k++)
                {
                        
                    //double mic = ((double)(client_msg->stream[k]) / 2097152.0); // / 65536.0; // / 2097152.0; // 2^21 65536.0; /// 33554432.0; // 65536.0; //16384.0; // / 16000.0;
                    rb->data[i + k] = (float)((double)(client_msg->stream[k]) / 262144); //  4194304.0);// / 32.0;
                    //rb->data[i + k] = (float)client_msg->stream[k]; //  / 16384.0; // mic;
                }
            }

            semop(semid, &my_sem_signal, 1);
            continue;

            //if (recv(socket_desc, client_msg, sizeof(msg), 0) < 0)
            //{
            //    printf("Couldn't receive\n");
            //    return -1;
            //}
            //// client_msg->stream[0] = client_msg->counter;
//
            //semop(semid, &my_sem_wait, 1);
//
            //// for (int i = 0; i < 1; i++)
            ////{
            ////     rb->data[rb->index] = (float)client_msg->stream[i];
            //// //
            ////    rb->index = (rb->index + 1) % BUFFER_LENGTH;
            ////}
//
            //// for (int i = 0; i < BUFFER_LENGTH; i+= N_SAMPLES)
//
            ////for (int i = 0; i < BUFFER_LENGTH; i+=N_MICROPHONES)
            ////{
            ////    for (int k = 0; k < N_MICROPHONES; k++)
            ////    {
////
            ////        //double mic = (double)(client_msg->stream[k]); // / 16000.0;
            ////        rb->data[i + k] = client_msg->stream[k]; //mic;
            ////    }
            ////}
//
            //// printf("%f\n", rb->data[0]);
//
            //write_buffer_int32(rb, client_msg->stream, N_MICROPHONES, 0);
//
            //rb->counter = rb->counter + 1;
            //// rb->data[rb->index] = (float)client_msg->counter;
//
            //// rb->index = (rb->index + 1) % BUFFER_LENGTH;
//
            //semop(semid, &my_sem_signal, 1);
        }

        free(client_msg);

        close_socket(socket_desc);
    }

    return 0;
}

int main(int argc, char const *argv[])
{
    //init_shared_memory();
//
    //for (int i = 0; i < BUFFER_LENGTH; i++) {
    //    rb->data[i] = 0.0;
    //}
//
    //init_semaphore();
//
    //pid_t pid = fork(); // Fork child
//
    //if (pid == -1)
    //{
    //    // printf("lonk\n");
    //    perror("fork");
    //    exit(1);
    //}
    //else if (pid == 0) // Child
    //{
    //    // Create UDP socket:
    //    int socket_desc = create_and_bind_socket();
//
    //    msg *client_msg = (msg *)calloc(1, sizeof(msg));
//
    //    //int counter = 0;
//
    //    while (1)
    //    {
    //        if (recv(socket_desc, client_msg, sizeof(msg), 0) < 0)
    //        {
    //            printf("Couldn't receive\n");
    //            return -1;
    //        }
    //        //client_msg->stream[0] = client_msg->counter;
//
    //        semop(semid, &my_sem_wait, 1);
//
    //        //for (int i = 0; i < 1; i++)
    //        //{
    //        //    rb->data[rb->index] = (float)client_msg->stream[i];
// ////
    //        //    rb->index = (rb->index + 1) % BUFFER_LENGTH;
    //        //}
//
    //        write_buffer_int32(rb, client_msg->stream, N_MICROPHONES, 0);
//
    //        rb->data[rb->index] = (float)client_msg->counter;
//
    //        // rb->index = (rb->index + 1) % BUFFER_LENGTH;
//
    //        semop(semid, &my_sem_signal, 1);
    //    }
//
    //    free(client_msg);
//
    //    close_socket(socket_desc);
    //}
    //else
    //{
    //    float out[BUFFER_LENGTH];
    //    while (1)
    //    {
    //        myread(&out[0]);
//
    //        //for (int i=0; i<BUFFER_LENGTH; i+=N_SAMPLES)
    //        //{
    //        //    printf("%f ", out[i]);
    //        //}
//
    //        printf("%f \n", out[0]);
    //        // sleep(1);
    //        //printf("%d\n", rb->index);
    //    }
    //}

    return 0;
}