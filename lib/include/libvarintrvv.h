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

#ifdef __cplusplus
}
#endif
#endif // LIBVARINTRVV_H