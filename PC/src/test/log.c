#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#define SHM_ID "/mmap-test"
#define BUFFER_SIZE 4096
#define SLEEP_NANOS 1000 // 1 micro

struct Message
{
    long _id;
    char _data[128];
};

struct RingBuffer
{
    size_t _rseq;
    char _pad1[64];

    size_t _wseq;
    char _pad2[64];

    Message _buffer[BUFFER_SIZE];
};

void producerLoop()
{
    int size = sizeof(RingBuffer);
    int fd = shm_open(SHM_ID, O_RDWR | O_CREAT, 0600);
    ftruncate(fd, size + 1);

    // create shared memory area
    RingBuffer *rb = (RingBuffer *)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    // initialize our sequence numbers in the ring buffer
    rb->_wseq = rb->_rseq = 0;
    int i = 0;

    timespec tss;
    tss.tv_sec = 0;
    tss.tv_nsec = SLEEP_NANOS;

    while (1)
    {
        // as long as the consumer isn't running behind keep producing
        while ((rb->_wseq + 1) % BUFFER_SIZE != rb->_rseq % BUFFER_SIZE)
        {
            // write the next entry and atomically update the write sequence number
            Message *msg = &rb->_buffer[rb->_wseq % BUFFER_SIZE];
            msg->_id = i++;
            __sync_fetch_and_add(&rb->_wseq, 1);
        }

        // give consumer some time to catch up
        nanosleep(&tss, 0);
    }
}

void consumerLoop()
{
    int size = sizeof(RingBuffer);
    int fd = shm_open(SHM_ID, O_RDWR, 0600);
    if (fd == -1)
    {
        perror("argh!!!");
        return;
    }

    // lookup producers shared memory area
    RingBuffer *rb = (RingBuffer *)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // initialize our sequence numbers in the ring buffer
    size_t seq = 0;
    size_t pid = -1;

    timespec tss;
    tss.tv_sec = 0;
    tss.tv_nsec = SLEEP_NANOS;

    while (1)
    {
        // while there is data to consume
        while (seq % BUFFER_SIZE != rb->_wseq % BUFFER_SIZE)
        {
            // get the next message and validate the id
            // id should only ever increase by 1
            // quit immediately if not
            Message msg = rb->_buffer[seq % BUFFER_SIZE];
            if (msg._id != pid + 1)
            {
                printf("error: %d %d\n", msg._id, pid);
                return;
            }
            pid = msg._id;
            ++seq;
        }

        // atomically update the read sequence in the ring buffer
        // making it visible to the producer
        __sync_lock_test_and_set(&rb->_rseq, seq);

        // wait for more data
        nanosleep(&tss, 0);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("please supply args (producer/consumer)\n");
        return -1;
    }
    else if (strcmp(argv[1], "consumer") == 0)
    {
        consumerLoop();
    }
    else if (strcmp(argv[1], "producer") == 0)
    {
        producerLoop();
    }
    else
    {
        printf("invalid arg: %s\n", argv[1]);
        return -1;
    }
}