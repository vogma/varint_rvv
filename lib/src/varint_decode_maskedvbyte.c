#include "libvarintrvv.h"
#include "utils.h"

static inline __attribute__((always_inline)) uint64_t create_mask(const vuint8m1_t varint_vec, const size_t vlmax_e8m1, const size_t length)
{
    vint8m1_t vec_i8 = __riscv_vreinterpret_v_u8m1_i8m1(varint_vec);

    vbool8_t mask = __riscv_vmslt_vx_i8m1_b8(vec_i8, 0, vlmax_e8m1);

    vuint8m1_t mask_as_u8 = __riscv_vreinterpret_v_b8_u8m1(mask);
    vuint64m1_t mask_as_u64 = __riscv_vreinterpret_v_u8m1_u64m1(mask_as_u8);
    return __riscv_vmv_x_s_u64m1_u64(mask_as_u64);
}

static inline __attribute__((always_inline)) uint64_t create_mask_m2(const vuint8m2_t varint_vec, const size_t vlmax_e8m2, const size_t length)
{
    vint8m2_t vec_i8 = __riscv_vreinterpret_v_u8m2_i8m2(varint_vec);

    vbool4_t mask = __riscv_vmslt_vx_i8m2_b4(vec_i8, 0, vlmax_e8m2);

    vuint32m1_t mask_as_u32 = __riscv_vreinterpret_v_b4_u32m1(mask);
    vuint64m1_t mask_as_u64 = __riscv_vreinterpret_v_u32m1_u64m1(mask_as_u32);
    return __riscv_vmv_x_s_u64m1_u64(mask_as_u64);
}

static inline __attribute__((always_inline)) uint64_t masked_vbyte_read_group(const vuint8m1_t in, uint32_t *out,
                                                                              uint64_t mask, uint64_t *ints_read, const size_t vlmax_e8m1)
{

    // fast path, all 16 bytes contain separate integers < 128
    if (__builtin_expect(!(mask & 0xFFFF), 1))
    {
        vuint32m4_t extended_result = __riscv_vzext_vf4_u32m4(in, vlmax_e8m1);
        __riscv_vse32_v_u32m4(out, extended_result, 16);
        *ints_read = 16;
        return 16;
    }

    const size_t vlmax_e16m1 = vlmax_e8m1 / 2;
    const size_t vlmax_e32m1 = vlmax_e8m1 / 4;
    const size_t vlmax_e64m1 = vlmax_e8m1 / 8;

    uint32_t low_12_bits = mask & 0xFFF;
    index_bytes_consumed combined = combined_lookup[low_12_bits];
    uint64_t consumed = combined.bytes_consumed;
    uint8_t index = combined.index;

    vint8m1_t vectors = __riscv_vle8_v_i8m1(&vectorsrawbytes[index * 16], 128 / 8);

    if (index < 64)
    {
        *ints_read = 6;
        vuint8m1_t shuffled = __riscv_vrgather_vv_u8m1(in, __riscv_vreinterpret_v_i8m1_u8m1(vectors), vlmax_e8m1);
        vuint16m1_t low_bytes = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_u8m1_u16m1(shuffled), 0x007F, vlmax_e16m1);
        vuint16m1_t high_bytes = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_u8m1_u16m1(shuffled), 0x7F00, vlmax_e16m1);
        vuint16m1_t high_bytes_shifted = __riscv_vsrl_vx_u16m1(high_bytes, 1, vlmax_e16m1);
        vuint16m1_t packed_result = __riscv_vor_vv_u16m1(high_bytes_shifted, low_bytes, vlmax_e16m1);
        vuint32m1_t unpacked_result_a = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_u16m1_u32m1(packed_result), 0x0000FFFF, vlmax_e32m1);
        __riscv_vse32_v_u32m1(out, unpacked_result_a, 4);
        vuint32m1_t unpacked_result_b = __riscv_vsrl_vx_u32m1(__riscv_vreinterpret_v_u16m1_u32m1(packed_result), 16, vlmax_e32m1);
        __riscv_vse32_v_u32m1(out + 4, unpacked_result_b, 2);
        return consumed;
    }

    if (index < 145)
    {
        *ints_read = 4;
        vuint8m1_t shuffled = __riscv_vrgather_vv_u8m1(in, __riscv_vreinterpret_v_i8m1_u8m1(vectors), vlmax_e8m1);
        vuint32m1_t shuffled32 = __riscv_vreinterpret_v_u8m1_u32m1(shuffled);
        vuint32m1_t low_bytes = __riscv_vand_vx_u32m1(shuffled32, 0x0000007F, vlmax_e32m1);
        vuint32m1_t middle_bytes = __riscv_vand_vx_u32m1(shuffled32, 0x00007F00, vlmax_e32m1);
        vuint32m1_t high_bytes = __riscv_vand_vx_u32m1(shuffled32, 0x007F0000, vlmax_e32m1);
        vuint32m1_t middle_bytes_shifted = __riscv_vsrl_vx_u32m1(middle_bytes, 1, vlmax_e32m1);
        vuint32m1_t high_bytes_shifted = __riscv_vsrl_vx_u32m1(high_bytes, 2, vlmax_e32m1);
        vuint32m1_t low_middle = __riscv_vor_vv_u32m1(low_bytes, middle_bytes_shifted, vlmax_e32m1);
        vuint32m1_t result = __riscv_vor_vv_u32m1(low_middle, high_bytes_shifted, vlmax_e32m1);
        __riscv_vse32_v_u32m1(out, result, 4);
        return consumed;
    }

    *ints_read = 2;

    vuint8m1_t data_bits = __riscv_vand_vx_u8m1(in, 0x7F, vlmax_e8m1);
    vuint8m1_t shuffled = __riscv_vrgather_vv_u8m1(data_bits, __riscv_vreinterpret_v_i8m1_u8m1(vectors), vlmax_e8m1);
    vuint64m1_t constant_vec = __riscv_vmv_v_x_u64m1(0x0010002000400080, vlmax_e64m1);
    vuint16m1_t split_bytes = __riscv_vmul_vv_u16m1(__riscv_vreinterpret_v_u8m1_u16m1(shuffled), __riscv_vreinterpret_v_u64m1_u16m1(constant_vec), vlmax_e16m1);
    vuint64m1_t shifted_split_bytes = __riscv_vsll_vx_u64m1(__riscv_vreinterpret_v_u16m1_u64m1(split_bytes), 8, vlmax_e64m1);
    vuint64m1_t recombined = __riscv_vor_vv_u64m1(shifted_split_bytes, __riscv_vreinterpret_v_u16m1_u64m1(split_bytes), vlmax_e64m1);
    vuint64m1_t low_byte = __riscv_vsrl_vx_u64m1(__riscv_vreinterpret_v_u8m1_u64m1(shuffled), 56, vlmax_e64m1);
    vuint64m1_t result_evens = __riscv_vor_vv_u64m1(recombined, low_byte, vlmax_e64m1);
    uint8_t gather_index[] = {0, 2, 4, 6, 8, 10, 12, 14, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    vuint8m1_t gather_vec = __riscv_vle8_v_u8m1(gather_index, vlmax_e8m1);
    vuint8m1_t result = __riscv_vrgather_vv_u8m1(__riscv_vreinterpret_v_u64m1_u8m1(result_evens), gather_vec, vlmax_e8m1);
    __riscv_vse32_v_u32m1(out, __riscv_vreinterpret_v_u8m1_u32m1(result), 2);

    return consumed;
}

size_t varint_decode_masked_vbyte(const uint8_t *input, size_t length, uint32_t *output)
{
    const size_t vlmax_e8m1 = 16; //__riscv_vsetvlmax_e8m1();
    uint64_t ints_read = 0;
    uint64_t ints_processed = 0;
    uint64_t consumed = 0;

    while (length >= 16)
    {
        const vuint8m1_t varint_vec = __riscv_vle8_v_u8m1(input, vlmax_e8m1);
        const uint64_t mask = create_mask(varint_vec, vlmax_e8m1, length);
        uint64_t consumed = masked_vbyte_read_group(varint_vec, output, mask, &ints_read, vlmax_e8m1);

        length -= consumed;
        input += consumed;
        output += ints_read;
        ints_processed += ints_read;
    }
    if (length > 0)
    {
        ints_processed += varint_decode_scalar(input, length, output);
    }

    return ints_processed;
}

size_t varint_decode_masked_vbyte_opt(const uint8_t *input, size_t length, uint32_t *output)
{
    const size_t vlmax_e8m2 = __riscv_vsetvlmax_e8m2();
    uint64_t ints_read = 0;
    uint64_t ints_processed = 0;
    uint64_t consumed = 0;

    while (length >= vlmax_e8m2)
    {

        vuint8m2_t varint_vec = __riscv_vle8_v_u8m2(input, vlmax_e8m2);
        uint64_t mask = create_mask_m2(varint_vec, vlmax_e8m2, length);

        size_t shifted = 0;

        if (mask == 0xFFFFFFFFFFFFFFFF)
        {
            vuint32m8_t result = __riscv_vzext_vf4_u32m8(varint_vec, vlmax_e8m2);
            __riscv_vse32_v_u32m8(output, result, vlmax_e8m2);
            ints_read += vlmax_e8m2;
            consumed += vlmax_e8m2;
        }
        else
        {
            while (shifted < (vlmax_e8m2 - 16))
            {
                uint64_t consumed = masked_vbyte_read_group(__riscv_vget_v_u8m2_u8m1(varint_vec, 0), output, mask, &ints_read, 16);
                varint_vec = __riscv_vslidedown_vx_u8m2(varint_vec, consumed, vlmax_e8m2);
                mask >>= consumed;
                shifted += consumed;
                length -= consumed;
                input += consumed;
                output += ints_read;
                ints_processed += ints_read;
            }
        }
    }
    if (length > 0)
    {
        ints_processed += varint_decode_scalar(input, length, output);
    }

    return ints_processed;
}
