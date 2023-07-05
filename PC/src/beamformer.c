#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

#include "config.h"
#include "circular_buffer.h"
#include "udp_receiver.h"
#include "antenna/antenna.h"
#include "antenna/delay.h"

ring_buffer *rb;  // Data to be stored in
msg *client_msg;

int shmid; // Shared memory ID
int semid; // Semaphore ID
int socket_desc;

struct sembuf sem_wait = {0, -1, SEM_UNDO};  // Wait operation
struct sembuf sem_signal = {0, 1, SEM_UNDO}; // Sig operation

float ****all_coefficients; // TODO fix this to more healthier

float **miso_coefficients;

/*
Remove shared memory and semafores
*/
void signal_handler()
{
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
Steer the antenna to the direction
*/
void steer(float theta, float phi)
{
    semop(semid, &sem_wait, 1);
    directional_antenna_delay_coefficients(theta, // Horizontal
                                           phi, // Vertical
                                           ROWS,
                                           COLUMNS,
                                           ELEMENT_DISTANCE, // Distance between elements
                                           SAMPLE_RATE,
                                           PROPAGATION_SPEED,
                                           miso_coefficients);
    semop(semid, &sem_signal, 1);
}


/*
Listen in one direction (WARNING must steer before usage)
*/
void miso(float *out)
{
    float data[BUFFER_LENGTH];

    // Pin the data for retrieval
    semop(semid, &sem_wait, 1);
    memcpy(&data[0], (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &sem_signal, 1);

    // Reset output stream
    memset(out, 0, (N_SAMPLES) * sizeof(float));

    for (int i = 0; i < ROWS * COLUMNS; i++)
    {
        // TODO Magic number since mic 1 does not work properly
        if (i == 1)
        {
            continue;
        }
        delay_vectorized_add(&data[i * N_SAMPLES], miso_coefficients[i], out);
    }

    for (int k = 0; k < N_SAMPLES; k++)
    {
        out[k] /= ROWS * COLUMNS - 1; // TODO Minus one to account for the faulty mic
    }
}

void mimo(float *image)
{
    float data[BUFFER_LENGTH];
    semop(semid, &sem_wait, 1);
    memcpy(&data[0], (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &sem_signal, 1);

    float _out[N_SAMPLES] = {0.0};
    float *out = &_out[0];

    // memcpy(miso, &data[0], sizeof(float) * N_SAMPLES);

    // printf("\x1b[H\x1b[J");

    for (int y = 0; y < MAX_RES; y++)
    {
        for (int x = 0; x < MAX_RES; x++)
        {

            memset(out, 0, (N_SAMPLES) * sizeof(float));

            // delay_vectorized_add(&data[0 * N_SAMPLES], all_coefficients[x][y][0], out);
            // delay_vectorized_add(&data[7 * N_SAMPLES], all_coefficients[x][y][7], out);
            // delay_vectorized_add(&data[55 * N_SAMPLES], all_coefficients[x][y][55], out);
            // delay_vectorized_add(&data[63 * N_SAMPLES], all_coefficients[x][y][63], out);
            for (int i = 0; i < 64; i++)
            {
                if (i == 1)
                {
                    continue;
                }
                // delay_vectorized_add(&data[i * N_SAMPLES], tmp_coefficients[i], out);
                delay_vectorized_add(&data[i * N_SAMPLES], all_coefficients[x][y][i], out);
            }
            float sum = 0.0;
            for (int k = 0; k < N_SAMPLES; k++)
            {
                out[k] /= 63.0;
                // out[k] /= 31.0;
                sum += powf((float)fabs((double)out[k]), 5);
            }

            sum /= (float)N_SAMPLES;

            image[y * MAX_RES + x] = sum;

            // if (sum > 2.0)
            //{
            //     printf("#");
            // } else {
            //     printf(" ");
            // }
        }

        // printf("\n");
    }

    // for (int x = 0; x < MAX_RES; x++)
    //{
    //     for (int y = 0; y < MAX_RES; y++)
    //     {
    //         memset(out, 0, (N_SAMPLES) * sizeof(float));
    //         for (int i = 2; i < 64; i++)
    //         {
    //             delay_vectorized_add(&data[i * N_SAMPLES], tmp_coefficients[i], out);
    //         }
    //         float sum = 0.0;
    //         for (int k = 0; k < N_SAMPLES; k++)
    //         {
    //             sum += powf((float)fabs((double)out[k] / 62), 2);
    //         }
    //
    //        //image[y * MAX_RES + x] = sum;
    //        printf("%d,%d = %f\n", x, y, sum);
    //    }
    //
    //}
}

/*
Get power-level in multiple directions
*/
void mimo2(float *image)
{
    float data[BUFFER_LENGTH];

    // Pin the data for retrieval
    semop(semid, &sem_wait, 1);
    memcpy(&data[0], (void *)&rb->data[0], sizeof(float) * BUFFER_LENGTH);
    semop(semid, &sem_signal, 1);

    // dummy output
    float _out[N_SAMPLES] = {0.0};
    float *out = &_out[0];

    // Start by scanning in Y-axis
    for (int y = 0; y < MAX_RES; y++)
    {
        for (int x = 0; x < MAX_RES; x++)
        {
            // Reset the output for the new direction
            memset(out, 0, (N_SAMPLES) * sizeof(float));

            for (int i = 0; i < N_MICROPHONES; i++)
            {
                // TODO mic 1 does not work
                if (i == 1)
                {
                    continue;
                }

                // Delay using the EPIC delay function ;)
                delay_vectorized_add(&data[i * N_SAMPLES], all_coefficients[x][y][i], out);
            }

            // Compute the power-level
            float sum = 0.0;
            for (int k = 0; k < N_SAMPLES; k++)
            {
                out[k] /= ROWS * COLUMNS - 1;
                sum += powf((float)fabs((double)out[k]), MISO_POWER);
            }

            sum /= (float)N_SAMPLES;

            image[y * MAX_RES + x] = sum;

            printf("SUM: %f\n", sum);
        }
    }
}

/*
TODO Fix too many nested mallocs
*/
void calculate_mimo_coefficients()
{
    all_coefficients = (float ****)malloc((MAX_RES) * sizeof(float ***));

    for (int x = 0; x < MAX_RES; x++)
    {
        all_coefficients[x] = (float ***)malloc(MAX_RES * sizeof(float **));

        for (int y = 0; y < MAX_RES; y++)
        {
            all_coefficients[x][y] = (float **)malloc((N_MICROPHONES) * sizeof(float *));

            for (int i = 0; i < N_MICROPHONES; i++)
            {
                all_coefficients[x][y][i] = (float *)malloc(N_TAPS * sizeof(float));
            }

            float horizontal = (x / (float)MAX_RES) * 2.0 * MAX_ANGLE - MAX_ANGLE;
            float vertical = (y / (float)MAX_RES) * 2.0 * MAX_ANGLE - MAX_ANGLE;

            directional_antenna_delay_coefficients(horizontal, // Horizontal
                                                   vertical,   // Vertical
                                                   ROWS,
                                                   COLUMNS,
                                                   ELEMENT_DISTANCE, // Distance between elements
                                                   SAMPLE_RATE,
                                                   PROPAGATION_SPEED,
                                                   all_coefficients[x][y]);
        }
    }
}

/*
Prepare coefficients
*/
void calculate_miso_coefficients()
{
    miso_coefficients = (float **)malloc((N_MICROPHONES) * sizeof(float *));

    for (int i = 0; i < N_MICROPHONES; i++)
    {
        miso_coefficients[i] = (float *)malloc(N_TAPS * sizeof(float));
    }

    directional_antenna_delay_coefficients(0.0, // Horizontal
                                           0.0,   // Vertical
                                           ROWS,
                                           COLUMNS,
                                           ELEMENT_DISTANCE, // Distance between elements
                                           SAMPLE_RATE,
                                           PROPAGATION_SPEED,
                                           miso_coefficients);
}

/*
Main initialization function
*/
int load()
{
    signal(SIGINT, signal_handler);
    signal(SIGKILL, signal_handler);
    signal(SIGTERM, signal_handler);

    init_shared_memory();
    init_semaphore();

    calculate_mimo_coefficients();
    calculate_miso_coefficients();

    printf("MISO: %f\n", miso_coefficients[0][0]);

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
            semop(semid, &sem_wait, 1);

            if (receive_and_write_to_buffer(socket_desc, rb, client_msg) == -1)
            {
                return -1;
            }

            semop(semid, &sem_signal, 1);
            continue;
        }
    }

    // Return to parent
    return 0;
}