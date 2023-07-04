/******************************************************************************
 * Title                 :   Perform MISO computation
 * Filename              :   miso.h
 * Author                :   Irreq
 * Origin Date           :   04/07/2023
 * Version               :   1.0.0
 * Compiler              :   gcc (GCC) 9.5.0
 * Target                :   x86_64 GNU/Linux
 * Notes                 :   None
 ******************************************************************************

 Functions to store scalar data efficiently in a circular buffer.

*/

#ifndef _MISO_H_
#define _MISO_H_

#include "config.h"

typedef struct _miso_ipc
{
    float x;
    float y;
    float data[N_SAMPLES];
} miso_ipc;

#endif