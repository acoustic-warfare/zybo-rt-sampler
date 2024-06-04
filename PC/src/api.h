#ifndef _API_H
#define _API_H

#include <stdbool.h>

int load(bool replay_mode);
int load_filter(bool replay_mode);
void get_data(float *signals);
void stop_receiving();
void signal_handler();

void pad_mimo(float *image, int *adaptive_array, int n);
void lerp_mimo(float *image, int *adaptive_array, int n);
void convolve_mimo_naive(float *image, int *adaptive_array, int n);
void convolve_mimo_vectorized(float *image, int *adaptive_array, int n);


void mimo_truncated(float *image, int *adaptive_array, int n);
void load_coefficients2(int *whole_samples, int n);

void miso_steer_listen(float *out, int *adaptive_array, int n, int steer_offset);
// void miso_steer_listen2(int *adaptive_array, int n, int steer_offset);

#include "portaudio.h"
#include "config.h"

typedef struct
{
    int can_read;
    float out[N_SAMPLES];
} paData;

typedef struct
{
    int steer_offset;
    float signals[BUFFER_LENGTH];
    int adaptive_array[N_MICROPHONES];
    int n;
} Miso;

int load_playback(paData *data);
int stop_playback();
int load_miso();
void load_pa(int *adaptive_array, int n);
void stop_miso();
void steer(int offset);

#endif