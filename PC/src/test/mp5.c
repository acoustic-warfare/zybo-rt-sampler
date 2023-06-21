/******************************************************************************
 * compile with gcc -pthread *.c -o loops
 * test with valgrind --tool=helgrind ./lops
 *
 ******************************************************************************/
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NTHREADS 4
#define ARRAYSIZE 100000000
#define ITERATIONS ARRAYSIZE / NTHREADS

double sum = 0.0;
double a[ARRAYSIZE];
pthread_mutex_t sum_mutex;

void *do_work(void *tid)
{
    int i, start, *mytid, end;
    double mysum = 0.0;

    /* Initialize my part of the global array and keep local sum */
    mytid = (int *)tid;
    start = (*mytid * ITERATIONS);
    end = start + ITERATIONS;
    printf("\n[Thread %5d] Doing iterations \t%10d to \t %10d", *mytid, start, end - 1);
    for (i = start; i < end; i++)
    {
        a[i] = i * 1.0;
        mysum = mysum + a[i];
    }

    /* Lock the mutex and update the global sum, then exit */
    pthread_mutex_lock(&sum_mutex);
    sum = sum + mysum;
    pthread_mutex_unlock(&sum_mutex);
    pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
    int i, start, tids[NTHREADS];
    pthread_t threads[NTHREADS];
    pthread_attr_t attr;

    /* Pthreads setup: initialize mutex and explicitly create threads in a
       joinable state (for portability).  Pass each thread its loop offset */
    pthread_mutex_init(&sum_mutex, NULL);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for (i = 0; i < NTHREADS; i++)
    {
        tids[i] = i;
        pthread_create(&threads[i], &attr, do_work, (void *)&tids[i]);
    }

    /* Wait for all threads to complete then print global sum */
    for (i = 0; i < NTHREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    printf("\n[MAIN] Done. Sum= %e", sum);

    sum = 0.0;
    /* for (i=0;i<ARRAYSIZE;i++){
     a[i] = i*1.0;
     sum = sum + a[i]; }
     printf("\n[MAIN] Check Sum= %e",sum);
   */
    /* Clean up and exit */
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&sum_mutex);
    pthread_exit(NULL);
}