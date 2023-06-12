# realtime-test


```bash
arm-none-eabi-gcc -lgcc -lc -lm demo.c --specs=nosys.specs -o demo.elf
arm-none-eabi-objcopy -O binary demo.out demo.bin
cp demo.bin /data/tftp/
```
