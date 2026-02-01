#ifndef LIBVARINTRVV_H
#define LIBVARINTRVV_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <stdio.h>
#include "riscv_vector.h"

    size_t varint_decode_m1(const uint8_t *input, size_t length, uint32_t *output);
    size_t varint_decode_m2(const uint8_t *input, size_t length, uint32_t *output);
    size_t varint_decode_masked_vbyte(const uint8_t *input, size_t length, uint32_t *output);
    size_t varint_decode_masked_vbyte_opt(const uint8_t *input, size_t length, uint32_t *output);
    size_t varint_decode_scalar(const uint8_t *input, int length, uint32_t *output);
    size_t vbyte_encode(const uint32_t *in, size_t length, uint8_t *bout);
    size_t varint_decode(const uint8_t *input, size_t length, uint32_t *output);
    size_t varint_decode_vecshift(const uint8_t *input, size_t length, uint32_t *output);
    size_t varint_decode_vecshift_m2(const uint8_t *input, size_t length, uint32_t *output);

#ifdef __cplusplus
}
#endif
#endif // LIBVARINTRVV_H