#include <stdio.h>
#include <stdlib.h>

#define IP_BASE 4

u32 Xil_In32(UINTPTR Addr)
{
    return *(volatile u32 *)Addr;
}

int hello() {
    int *p = (int *)0x40000000;

    return 0;
}


int read_fpga()
{

}

/**
Method for writing directly to memory region
*/
void write_dw_devmem2(int base_address, int offset, int value) {
    int addr = base_address + offset;
}

/**
Method for reading from memory region
*/
int read_dw_devmem2(int base_address, int offset)
{
    return 0;
}

int main(int argc, char *argv[])
{

    FILE *fp;
    char path[1035];

    /* Open the command for reading. */
    fp = popen("/bin/ls /etc/", "r");
    if (fp == NULL)
    {
        printf("Failed to run command\n");
        exit(1);
    }

    /* Read the output a line at a time - output it. */
    while (fgets(path, sizeof(path), fp) != NULL)
    {
        printf("%s", path);
    }

    /* close */
    pclose(fp);

    return 0;
}