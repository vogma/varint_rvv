#include <libvarintrvv.h>

// source https://chromium.googlesource.com/external/github.com/google/protobuf/%2B/refs/heads/master/src/google/protobuf/io/coded_stream.cc#366
inline __attribute__((always_inline)) const size_t ReadVarint32FromArray(const uint8_t *buffer, uint32_t *value)
{
    size_t bytes_processed = 0;
    const uint8_t *ptr = buffer;
    uint32_t first = *ptr++;

    if (first < 128)
    {
        *value = first;
        return 1;
    }

    uint32_t b;
    uint32_t result = first - 0x80;

    b = *ptr++;
    result += b << 7;
    if (!(b & 0x80))
    {
        bytes_processed = 2;
        goto done;
    }
    result -= 0x80 << 7;
    b = *ptr++;
    result += b << 14;
    if (!(b & 0x80))
    {
        bytes_processed = 3;
        goto done;
    }
    result -= 0x80 << 14;
    b = *ptr++;
    result += b << 21;
    if (!(b & 0x80))
    {
        bytes_processed = 4;
        goto done;
    }
    result -= 0x80 << 21;
    b = *ptr++;
    result += b << 28;
    if (!(b & 0x80))
    {
        bytes_processed = 5;
        goto done;
    }
done:
    *value = result;
    return bytes_processed;
}

size_t varint_decode_scalar(const uint8_t *input, int length, uint32_t *output)
{
    uint32_t *out = output;
    while (length > 0)
    {
        size_t bytes_processed = ReadVarint32FromArray(input, out);
        length -= bytes_processed;
        input += bytes_processed;
        out++;
    }
    return out - output;
}
