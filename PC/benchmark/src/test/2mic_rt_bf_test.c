// gcc -o run 2mic_rt_bf_test.c -lm -lrt -lasound -ljack -lpthread -lportaudio && ./run
#include <stdio.h>
#include <math.h>
#include "portaudio.h"

#define NUM_SECONDS (10)
#define SAMPLE_RATE (48828)
#define FRAMES_PER_BUFFER (1)

#ifndef M_PI
#define M_PI (3.14159265)
#endif

#define TABLE_SIZE (200)
typedef struct
{
    float sine[TABLE_SIZE];
    int left_phase;
    int right_phase;
    char message[20];
} paTestData;

/* This routine will be called by the PortAudio engine when audio is needed.
** It may called at interrupt level on some machines so don't do anything
** that could mess up the system like calling malloc() or free().
*/
static int patestCallback(const void *inputBuffer, void *outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo *timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void *userData)
{
    (void)userData;
    float *out = (float *)outputBuffer;
    unsigned long i;
    float *in = (float *)inputBuffer;

    float sum = 0.0;

    for (i = 0; i < framesPerBuffer; i++)
    {
        sum = *in++;
        sum += *in++;
        sum /= 2.0;
        *out++ = sum;
        *out++ = sum;
    }

    return paContinue;
}

/*
 * This routine is called by portaudio when playback is done.
 */
static void StreamFinished(void *userData)
{
    paTestData *data = (paTestData *)userData;
    printf("Stream Completed: %s\n", data->message);
}

/*******************************************************************/
int main(void);
int main(void)
{
    PaStreamParameters outputParameters, inputParameters;
    PaStream *stream;
    PaError err;
    paTestData data;
    int i;

    printf("PortAudio Test: output sine wave. SR = %d, BufSize = %d\n", SAMPLE_RATE, FRAMES_PER_BUFFER);

    /* initialise sinusoidal wavetable */
    for (i = 0; i < TABLE_SIZE; i++)
    {
        data.sine[i] = (float)sin(((double)i / (double)TABLE_SIZE) * M_PI * 2.);
    }
    data.left_phase = data.right_phase = 0;

    err = Pa_Initialize();
    if (err != paNoError)
        goto error;

    outputParameters.device = Pa_GetDefaultOutputDevice(); /* default output device */
    if (outputParameters.device == paNoDevice)
    {
        fprintf(stderr, "Error: No default output device.\n");
        goto error;
    }
    outputParameters.channelCount = 2;         /* stereo output */
    outputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
    outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = NULL;

    inputParameters.device = Pa_GetDefaultInputDevice(); /* default output device */
    if (inputParameters.device == paNoDevice)
    {
        fprintf(stderr, "Error: No default output device.\n");
        goto error;
    }
    inputParameters.channelCount = 2;         /* stereo output */
    inputParameters.sampleFormat = paFloat32; /* 32 bit floating point output */
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;

    err = Pa_OpenStream(
        &stream,
        &inputParameters, /* no input */
        &outputParameters,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paClipOff, /* we won't output out of range samples so don't bother clipping them */
        patestCallback,
        &data);
    if (err != paNoError)
        goto error;

    sprintf(data.message, "No Message");
    err = Pa_SetStreamFinishedCallback(stream, &StreamFinished);
    if (err != paNoError)
        goto error;

    err = Pa_StartStream(stream);
    if (err != paNoError)
        goto error;

    printf("Play for %d seconds.\n", NUM_SECONDS);
    Pa_Sleep(NUM_SECONDS * 1000);

    err = Pa_StopStream(stream);
    if (err != paNoError)
        goto error;

    err = Pa_CloseStream(stream);
    if (err != paNoError)
        goto error;

    Pa_Terminate();
    printf("Test finished.\n");

    return err;
error:
    Pa_Terminate();
    fprintf(stderr, "An error occurred while using the portaudio stream\n");
    fprintf(stderr, "Error number: %d\n", err);
    fprintf(stderr, "Error message: %s\n", Pa_GetErrorText(err));
    return err;
}