#ifndef DELAY_H
#define DELAY_H
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
#include <stdbool.h>
#include <unistd.h>
#include <signal.h>

#include "beamformer.h"
#include "config.h"
#include "circular_buffer.h"
#include "udp_receiver.h"
#include "antenna/antenna.h"
#include "antenna/delay.h"
int load(bool replay_mode);
//void delay_truncation_sum(float *signal, float *out, int n);
//void bar(float *signal);
void kill_child();
void myread(float *out);
void miso(float *out);
void steer(float theta, float phi);
void mimo_truncated(float *image, int *adaptive_array, int n);
#endif
