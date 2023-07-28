#include <stdio.h>
#include <math.h>
#include "portaudio.h"
#include "config.h"
#include "playback.h"

#define DEBUG_PLAYBACK 0

PaStreamParameters outputParameters, inputParameters;
PaStream *stream;
PaError err;

/* This routine will be called by the PortAudio engine when audio is needed.
** It may called at interrupt level on some machines so don't do anything
** that could mess up the system like calling malloc() or free().
*/
static int playback_callback(const void *inputBuffer, void *outputBuffer,
                             unsigned long framesPerBuffer,
                             const PaStreamCallbackTimeInfo *timeInfo,
                             PaStreamCallbackFlags statusFlags,
                             void *userData)
{
    paData *data = (paData *)userData;
    float *out = (float *)outputBuffer;
    (void)inputBuffer;

    unsigned long i;

    // while (!data->can_read)
    // {
    //     ;
    // }

    // data->can_read = 0;
    // for (i = 0; i < N_SAMPLES; i++)
    // {
    //     *out++ = data->out[i];
    //     *out++ = data->out[i];
    // }

    if (data->can_read)
    {
        data->can_read = 0;
        for (i = 0; i < N_SAMPLES; i++)
        {
            *out++ = data->out[i];
            *out++ = data->out[i];
        }
        
    }
    
    return paContinue;
}

/**
 * @brief Stop the current playback audio stream
 * 
 * @return int 
 */
int stop_playback()
{
    err = Pa_StopStream(stream);
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));

    err = Pa_CloseStream(stream);
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));

    Pa_Terminate();

    return err;
}

/*
 * This routine is called by portaudio when playback is done.
 */
static void StreamFinished(void *userData)
{
    printf("Stream completed\n");
}

int load_playback(paData *data)
{
    err = Pa_Initialize();
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));

    outputParameters.device = Pa_GetDefaultOutputDevice(); /* default output device */
    if (outputParameters.device == paNoDevice)
    {
        fprintf(stderr, "Error: No default output device.\n");
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
    }
    outputParameters.channelCount = 2;         /* stereo output */
    outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
    outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = NULL;

    err = Pa_OpenStream(
        &stream,
        NULL, /* no input */
        &outputParameters,
        SAMPLE_RATE,
        N_SAMPLES,
        paClipOff, /* we won't output out of range samples so don't bother clipping them */
        playback_callback,
        data);
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));

    err = Pa_SetStreamFinishedCallback(stream, &StreamFinished);
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));

    err = Pa_StartStream(stream);
    if (err != paNoError)
        fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));

    printf("Streaming data in the background\n");
}

#if DEBUG_PLAYBACK
int main(int argc, char const *argv[])
{
    paData data;
    return 0;
}

#endif