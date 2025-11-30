#include "libvarintrvv.h"

uint32_t varint_decode(uint8_t *input, uint32_t *output, size_t length)
{
    const size_t vlmax_e8m1 = __riscv_vsetvlmax_e8m1();

    const vuint8m1_t varint_vec = __riscv_vle8_v_u8m1(input, vlmax_e8m1);

    // convert vuint32m1_t -> vuint8m1_t -> vint8m1_t
    vint8m1_t vec_i8 = __riscv_vreinterpret_v_u8m1_i8m1(varint_vec);

    // signed comparison with 0 checks if MSB is set
    vbool8_t mask = __riscv_vmslt_vx_i8m1_b8(vec_i8, 0, vlmax_e8m1);

    vuint8m1_t mask_as_u8 = __riscv_vreinterpret_v_b8_u8m1(mask);
    vuint32m1_t mask_as_u32 = __riscv_vreinterpret_v_u8m1_u32m1(mask_as_u8);
    return __riscv_vmv_x_s_u32m1_u32(mask_as_u32);
}