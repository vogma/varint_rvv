#ifndef LIBVARINTRVV_H
#define LIBVARINTRVV_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "stdio.h"
#include "stdint.h"
#include "stdlib.h"
#include "riscv_vector.h"

    uint64_t varint_decode(uint8_t *input, uint32_t *output, size_t length);
    uint64_t masked_vbyte_read_group(const vuint8m1_t in, uint32_t *out, uint64_t mask, uint64_t *ints_read, const size_t vlmax_e8m1);
#ifdef __cplusplus
}
#endif
#endif // LIBVARINTRVV_H