#ifndef _PLAYBACK_H_
#define _PLAYBACK_H_

#include "config.h"
#include "portaudio.h"

typedef struct
{
    int can_read;
    float out[N_SAMPLES];
} paData;

int load_playback(paData *data);
int stop_playback();
#endif