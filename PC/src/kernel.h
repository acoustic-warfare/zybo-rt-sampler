#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif
#include <cufft.h>

EXTERNC void  cuda_filter_init();
EXTERNC void plan_init();
EXTERNC void memory_init();
EXTERNC void cuda_init_all();
EXTERNC void cuda_filter_FFT(float *input, float *filter_output);

