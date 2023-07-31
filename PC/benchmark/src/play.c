#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>

#include "api.h"
#include "playback.h"
#include "play.h"
#include "config.h"
paData data;

#include "algorithms/pad_and_sum.h"
// #include "algorithms/convolve_and_sum.h"

#include <unistd.h>
#include <stdio.h>
#include <signal.h>

#define NMICS N_MICROPHONES

volatile sig_atomic_t stop;


#include <stdio.h>
/**
 * @brief Create a stream for portaudio
 *
 */
void init_portaudio_playback()
{
    data.can_read = 0;
    for (int i = 0; i < N_SAMPLES; i++)
    {
        data.out[i] = 0.0;
    }

    load_playback(&data);
}

void loop()
{
    int coefficients[NMICS] = {0};
    load_coefficients_pad(&coefficients[0], NMICS);

    int adaptive_array[NMICS];
    int n = NMICS;

    for (int i = 0; i < n; i++)
    {
        adaptive_array[i] = i;
    }

    float signals[BUFFER_LENGTH];

    paData *d = &data;

    int num = 150;

    float factor = 1.0;

    float signal[N_SAMPLES] = {0.0};

    while (!stop)
    {
        int k;
        get_data(&signals[0]);
        miso_pad(&signals[0], &signal[0], &adaptive_array[0], n, 0);

        for (int i = 0; i < N_SAMPLES; i++)
        {
            d->out[i] = signal[i] * factor;
        }
        d->can_read = 1;
    }
}

void loop2(int *adaptive_array, int n)
{
    int coefficients[NMICS] = {0};
    load_coefficients_pad(&coefficients[0], NMICS);

    float signals[BUFFER_LENGTH];

    paData *d = &data;

    int num = 150;

    float factor = 1.0;

    float signal[N_SAMPLES] = {0.0};

    while (!stop)
    {
        int k;
        get_data(&signals[0]);
        miso_pad(&signals[0], &signal[0], adaptive_array, n, 0);

        for (int i = 0; i < N_SAMPLES; i++)
        {
            d->out[i] = signal[i] * factor;
        }
        d->can_read = 1;
    }
}

void inthand(int signum)
{
    stop = 1;
}

// int main(int argc, char **argv)
// {

//     signal(SIGINT, inthand);

//     init_portaudio_playback();
//     load(false);
//     printf("Connected\n");
//     loop();

//     printf("exiting safely\n");
//     // system("pause");

//     stop_receiving();
//     signal_handler();

//     return 0;
// }


int start_mic()
{
    signal(SIGINT, inthand);

    init_portaudio_playback();
    load(false);
    printf("Connected\n");
    loop();

    printf("exiting safely\n");
    // system("pause");

    stop_receiving();
    signal_handler();
}

int start_mic2(int *adaptive_array, int n)
{
    signal(SIGINT, inthand);

    init_portaudio_playback();
    load(false);
    printf("Connected\n");
    loop2(adaptive_array, n);

    printf("exiting safely\n");
    // system("pause");

    stop_receiving();
    signal_handler();
}

// int main(int argc, char const *argv[])
// {
//     init_portaudio_playback();
//     load(false);
//     printf("Connected\n");
//     loop();

    
//     stop_receiving();
//     signal_handler();
//     return 0;
// }
