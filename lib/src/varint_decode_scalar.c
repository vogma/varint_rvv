#include <libvarintrvv.h>

static int read_int(const uint8_t *in, uint32_t *out)
{
    *out = in[0] & 0x7F;
    if (in[0] < 128)
    {
        return 1;
    }
    *out = ((in[1] & 0x7FU) << 7) | *out;
    if (in[1] < 128)
    {
        return 2;
    }
    *out = ((in[2] & 0x7FU) << 14) | *out;
    if (in[2] < 128)
    {
        return 3;
    }
    *out = ((in[3] & 0x7FU) << 21) | *out;
    if (in[3] < 128)
    {
        return 4;
    }
    *out = ((in[4] & 0x7FU) << 28) | *out;
    return 5;
}

size_t varint_decode_scalar(const uint8_t *input, int length, uint32_t *output)
{
    uint32_t *output_orig = output;
    while (length > 0)
    {
        size_t bytes_processed = read_int(input, output);
        length -= bytes_processed;
        input += bytes_processed;
        output++;
    }
    return output - output_orig;
}
