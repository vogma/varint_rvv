#include "libvarintrvv.h"
#include "utils.h"

uint32_t varint_decode(uint8_t *input, uint32_t *output, size_t length)
{
    const size_t vlmax_e8m1 = __riscv_vsetvlmax_e8m1();
    const size_t vlmax_e16m1 = __riscv_vsetvlmax_e16m1();
    const size_t vlmax_e32m1 = __riscv_vsetvlmax_e32m1();

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

    vint8m1_t vectors = __riscv_vle8_v_i8m1(&vectorsrawbytes[index*16], 128/8);

    uint8_t ints_read = 0;

    if (index < 64)
    {
        ints_read = 6;

        // uint8_t debug_vals[16];

        // __riscv_vse8_v_i8m1(debug_vals,  vectors, vlmax_e8m1);
        // printf("index: %02x %02x %02x %02x %02x %02x %02x %02x\n", debug_vals[0], debug_vals[1], debug_vals[2], debug_vals[3], debug_vals[4], debug_vals[5], debug_vals[6], debug_vals[7]);

        vuint8m1_t shuffled = __riscv_vrgather_vv_u8m1(varint_vec, __riscv_vreinterpret_v_i8m1_u8m1(vectors), vlmax_e8m1);

        // __riscv_vse8_v_u8m1(debug_vals, shuffled, vlmax_e8m1);
        // printf("shuffled: %02x %02x %02x %02x\n", debug_vals[0], debug_vals[1], debug_vals[2], debug_vals[3]);

        vuint16m1_t low_bytes = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_u8m1_u16m1(shuffled), 0x007F, vlmax_e16m1);
        vuint16m1_t high_bytes = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_u8m1_u16m1(shuffled), 0x7F00, vlmax_e16m1);

        vuint16m1_t high_bytes_shifted = __riscv_vsrl_vx_u16m1(high_bytes, 1, vlmax_e16m1);
        vuint16m1_t packed_result = __riscv_vor_vv_u16m1(high_bytes_shifted, low_bytes, vlmax_e16m1);

        vuint32m1_t unpacked_result_a = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_u16m1_u32m1(packed_result), 0x0000FFFF, vlmax_e32m1);

        __riscv_vse32_v_u32m1(output, unpacked_result_a, 4);


        vuint32m1_t unpacked_result_b = __riscv_vsrl_vx_u32m1(__riscv_vreinterpret_v_u16m1_u32m1(packed_result), 16, vlmax_e32m1);
        __riscv_vse32_v_u32m1(output+4, unpacked_result_b, 2);

        		// __m128i unpacked_result_b = _mm_srli_epi32(packed_result, 16);
		// _mm_storel_epi64(mout+1, unpacked_result_b);

        // __m128i high_bytes_shifted = _mm_srli_epi16(high_bytes, 1);
        // __m128i packed_result = _mm_or_si128(low_bytes, high_bytes_shifted);
        // __m128i unpacked_result_a = _mm_and_si128(packed_result,
        // _mm_set1_epi32(0x0000FFFF));
        // _mm_storeu_si128(mout, unpacked_result_a);
    }

    return (uint32_t)result;
}