#include "libvarintrvv.h"
#include "utils.h"

uint32_t varint_decode(uint8_t *input, uint32_t *output, size_t length)
{
    const size_t vlmax_e8m1 = __riscv_vsetvlmax_e8m1();

    const vuint8m1_t varint_vec = __riscv_vle8_v_u8m1(input, vlmax_e8m1);

    vint8m1_t vec_i8 = __riscv_vreinterpret_v_u8m1_i8m1(varint_vec);

    vbool8_t mask = __riscv_vmslt_vx_i8m1_b8(vec_i8, 0, vlmax_e8m1);

    vuint8m1_t mask_as_u8 = __riscv_vreinterpret_v_b8_u8m1(mask);
    vuint64m1_t mask_as_u64 = __riscv_vreinterpret_v_u8m1_u64m1(mask_as_u8);
    uint64_t result = __riscv_vmv_x_s_u64m1_u64(mask_as_u64);

    uint32_t low_12_bits = result & 0xFFF;
    // combine index and bytes consumed into a single lookup
    index_bytes_consumed combined = combined_lookup[low_12_bits];
    uint64_t consumed = combined.bytes_consumed;
    uint8_t index = combined.index;

    printf("consumed %ld index %d\n", consumed, index);

    vuint8m1_t vectors = __riscv_vle8_v_u8m1(&vectorsrawbytes[index], vlmax_e8m1);

    return (uint32_t)result;
}