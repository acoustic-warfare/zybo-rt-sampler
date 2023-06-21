#pragma pack(1)

// Version 1.0 (BatMobile 1000)
typedef struct payload_protocol_t
{
    int id;               // Transfer id, for tracking later
    int protocol_version; // To trace error
    int fs;               // Sampling rate
    int fs_nr;            // Sample number
    int samples;          // Every mic
    int sample_error;     // If error inside
    int bitstream[192];   // The bitstream from the mic array
} payload_protocol;

#pragma pack()