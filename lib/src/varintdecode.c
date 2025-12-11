#include "libvarintrvv.h"
#include "utils.h"

/**
 * Extract the MSB of all Bytes in data and return a scalar mask.
 */
static inline uint64_t createMask(vint8m1_t data, const size_t vl)
{
    vbool8_t mask = __riscv_vmslt_vx_i8m1_b8(data, 0, vl);                                             // a comparison < 0 with the data interpreted as a signed integer gives us a mask with all bytes which have the MSB set
    vuint64m1_t mask_as_u64 = __riscv_vreinterpret_v_u8m1_u64m1(__riscv_vreinterpret_v_b8_u8m1(mask)); // Type conversion: b8 -> u8 -> u64
    return __riscv_vmv_x_s_u64m1_u64(mask_as_u64);                                                     // extract first element into scalar register
}

static inline uint64_t createMaskM8(vint8m8_t data, const size_t vl)
{
    vbool1_t mask = __riscv_vmslt_vx_i8m8_b1(data, 0, vl);                                             // a comparison < 0 with the data interpreted as a signed integer gives us a mask with all bytes which have the MSB set
    vuint64m1_t mask_as_u64 = __riscv_vreinterpret_v_u8m1_u64m1(__riscv_vreinterpret_v_b1_u8m1(mask)); // Type conversion: b8 -> u8 -> u64
    return __riscv_vmv_x_s_u64m1_u64(mask_as_u64);                                                     // extract first element into scalar register
}

// static uint64_t masked_vbyte_read_group(const vuint8m1_t in, uint32_t *out,
//                                         uint64_t mask, uint64_t *ints_read, const size_t vlmax_e8m1)
// {

//     // fast path, all 16 bytes contain separate integers < 128
//     if (!(mask & 0xFFFF))
//     {
//         vuint32m4_t extended_result = __riscv_vzext_vf4_u32m4(in, vlmax_e8m1);
//         __riscv_vse32_v_u32m4(out, extended_result, 16);
//         *ints_read = 16;
//         return 16;
//     }

//     const size_t vlmax_e16m1 = vlmax_e8m1 / 2;
//     const size_t vlmax_e32m1 = vlmax_e8m1 / 4;
//     const size_t vlmax_e64m1 = vlmax_e8m1 / 8;

//     uint32_t low_12_bits = mask & 0xFFF;
//     index_bytes_consumed combined = combined_lookup[low_12_bits];
//     uint64_t consumed = combined.bytes_consumed;
//     uint8_t index = combined.index;

//     vint8m1_t vectors = __riscv_vle8_v_i8m1(&vectorsrawbytes[index * 16], 128 / 8);

//     if (index < 64)
//     {
//         *ints_read = 6;
//         vuint8m1_t shuffled = __riscv_vrgather_vv_u8m1(in, __riscv_vreinterpret_v_i8m1_u8m1(vectors), vlmax_e8m1);
//         vuint16m1_t low_bytes = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_u8m1_u16m1(shuffled), 0x007F, vlmax_e16m1);
//         vuint16m1_t high_bytes = __riscv_vand_vx_u16m1(__riscv_vreinterpret_v_u8m1_u16m1(shuffled), 0x7F00, vlmax_e16m1);
//         vuint16m1_t high_bytes_shifted = __riscv_vsrl_vx_u16m1(high_bytes, 1, vlmax_e16m1);
//         vuint16m1_t packed_result = __riscv_vor_vv_u16m1(high_bytes_shifted, low_bytes, vlmax_e16m1);
//         vuint32m1_t unpacked_result_a = __riscv_vand_vx_u32m1(__riscv_vreinterpret_v_u16m1_u32m1(packed_result), 0x0000FFFF, vlmax_e32m1);
//         __riscv_vse32_v_u32m1(out, unpacked_result_a, 4);
//         vuint32m1_t unpacked_result_b = __riscv_vsrl_vx_u32m1(__riscv_vreinterpret_v_u16m1_u32m1(packed_result), 16, vlmax_e32m1);
//         __riscv_vse32_v_u32m1(out + 4, unpacked_result_b, 2);
//         return consumed;
//     }

//     if (index < 145)
//     {
//         *ints_read = 4;
//         vuint8m1_t shuffled = __riscv_vrgather_vv_u8m1(in, __riscv_vreinterpret_v_i8m1_u8m1(vectors), vlmax_e8m1);
//         vuint32m1_t shuffled32 = __riscv_vreinterpret_v_u8m1_u32m1(shuffled);
//         vuint32m1_t low_bytes = __riscv_vand_vx_u32m1(shuffled32, 0x0000007F, vlmax_e32m1);
//         vuint32m1_t middle_bytes = __riscv_vand_vx_u32m1(shuffled32, 0x00007F00, vlmax_e32m1);
//         vuint32m1_t high_bytes = __riscv_vand_vx_u32m1(shuffled32, 0x007F0000, vlmax_e32m1);
//         vuint32m1_t middle_bytes_shifted = __riscv_vsrl_vx_u32m1(middle_bytes, 1, vlmax_e32m1);
//         vuint32m1_t high_bytes_shifted = __riscv_vsrl_vx_u32m1(high_bytes, 2, vlmax_e32m1);
//         vuint32m1_t low_middle = __riscv_vor_vv_u32m1(low_bytes, middle_bytes_shifted, vlmax_e32m1);
//         vuint32m1_t result = __riscv_vor_vv_u32m1(low_middle, high_bytes_shifted, vlmax_e32m1);
//         __riscv_vse32_v_u32m1(out, result, 4);
//         return consumed;
//     }

//     *ints_read = 2;

//     vuint8m1_t data_bits = __riscv_vand_vx_u8m1(in, 0x7F, vlmax_e8m1);
//     vuint8m1_t shuffled = __riscv_vrgather_vv_u8m1(data_bits, __riscv_vreinterpret_v_i8m1_u8m1(vectors), vlmax_e8m1);
//     vuint64m1_t constant_vec = __riscv_vmv_v_x_u64m1(0x0010002000400080, vlmax_e64m1);
//     vuint16m1_t split_bytes = __riscv_vmul_vv_u16m1(__riscv_vreinterpret_v_u8m1_u16m1(shuffled), __riscv_vreinterpret_v_u64m1_u16m1(constant_vec), vlmax_e16m1);
//     vuint64m1_t shifted_split_bytes = __riscv_vsll_vx_u64m1(__riscv_vreinterpret_v_u16m1_u64m1(split_bytes), 8, vlmax_e64m1);
//     vuint64m1_t recombined = __riscv_vor_vv_u64m1(shifted_split_bytes, __riscv_vreinterpret_v_u16m1_u64m1(split_bytes), vlmax_e64m1);
//     vuint64m1_t low_byte = __riscv_vsrl_vx_u64m1(__riscv_vreinterpret_v_u8m1_u64m1(shuffled), 56, vlmax_e64m1);
//     vuint64m1_t result_evens = __riscv_vor_vv_u64m1(recombined, low_byte, vlmax_e64m1);
//     uint8_t gather_index[] = {0, 2, 4, 6, 8, 10, 12, 14, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
//     vuint8m1_t gather_vec = __riscv_vle8_v_u8m1(gather_index, vlmax_e8m1);
//     vuint8m1_t result = __riscv_vrgather_vv_u8m1(__riscv_vreinterpret_v_u64m1_u8m1(result_evens), gather_vec, vlmax_e8m1);
//     __riscv_vse32_v_u32m1(out, __riscv_vreinterpret_v_u8m1_u32m1(result), 2);

//     return consumed;
// }

vuint8m1_t scatter_via_gather(vuint8m1_t data, vbool8_t target_mask, size_t vl)
{
    // target_mask has bits 1 and 2 set.

    // 1. Calculate which source index belongs to which destination lane
    // Dest Lane 0 (Mask 0) -> result irrelevant
    // Dest Lane 1 (Mask 1) -> 0 set bits before -> Index 0
    // Dest Lane 2 (Mask 1) -> 1 set bit before -> Index 1
    vuint8m1_t gather_indices = __riscv_viota_m_u8m1(target_mask, vl);

    // 2. Initialize result (e.g. to 0x00)
    vuint8m1_t result = __riscv_vmv_v_x_u8m1(0, vl);

    // 3. Pull the data into place
    // "For every lane i where mask is 1, look at data[gather_indices[i]]"
    result = __riscv_vrgather_vv_u8m1_m(target_mask, data, gather_indices, vl);

    return result;
}

static inline uint64_t createSecondBytesMask(vint8m1_t data, size_t vl)
{
    uint64_t masku8 = createMask(data, vl);
    uint64_t mask_shifted = masku8 << 1;
    mask_shifted = ~mask_shifted;
    return masku8 & mask_shifted;
}

/**
 * Number of varints that will be processed in the current loop iteration.
 */
static inline uint8_t getNumberOfVarints(vbool8_t varint_mask, size_t vl)
{
    return __riscv_vcpop_m_b8(varint_mask, vl);
}

/**
 * It's possible that the register splices a varint in half. So we can only decode complete varints in each vec.
 * This function returns the number of bytes that are occupied by complete varints in the vector register.
 */
static inline uint8_t getCompleteVarintSize(vint8m1_t data, vbool8_t varint_mask, size_t vl)
{
    // every lane gets an index
    vuint8m1_t index_vec = __riscv_vid_v_u8m1(vl);

    // set every byte which has the continuation bit set to zero
    vuint8m1_t index_of_varints = __riscv_vmerge_vxm_u8m1(index_vec, 0, varint_mask, vl);

    // find the largest index value and place in the lowest lane
    index_of_varints = __riscv_vredmax_vs_u8m1_u8m1(index_of_varints, __riscv_vmv_v_x_u8m1(0, vl), vl);

    // return lowest lane in scalar reg (+1 to convert Index (0-based) to Count (1-based))
    return __riscv_vmv_x_s_u8m1_u8(index_of_varints) + 1;
}

uint64_t varint_decode(uint8_t *input, uint32_t *output, size_t length)
{
    const size_t vlmax_e8m1 = __riscv_vsetvlmax_e8m1();
    const size_t vlmax_e64m1 = vlmax_e8m1 / 8;

    // vuint8m1_t varint_vec = __riscv_vle8_v_u8m1(input, vlmax_e8m1);

    // vint8m1_t vec_i8 = __riscv_vreinterpret_v_u8m1_i8m1(varint_vec);

    // uint64_t mask = createMask(vec_i8, vlmax_e8m1);

    uint64_t ints_read;

    uint8_t data[16] = {0x08, 0x84, 0x02, 0x81, 0x03, 0x08, 0x84, 0x02, 0x81, 0x03, 0x08, 0x84, 0x02, 0x81, 0x83, 0x80};

    vint8m1_t data_vec = __riscv_vle8_v_i8m1(data, vlmax_e8m1);

    vbool8_t mask = __riscv_vmslt_vx_i8m1_b8(data_vec, 0, vlmax_e8m1); // a comparison < 0 with the data interpreted as a signed integer gives a mask with all bytes which have the MSB set

    vbool8_t inverted_mask = __riscv_vreinterpret_v_u8m1_b8(__riscv_vnot_v_u8m1(__riscv_vreinterpret_v_b8_u8m1(mask), vlmax_e8m1));

    uint8_t number_of_bytes = getCompleteVarintSize(data_vec, mask, vlmax_e8m1);

    printf("Number of Bytes: %d\n", number_of_bytes);

    uint8_t number_of_varints = getNumberOfVarints(mask, vlmax_e8m1);

    printf("Number of Varints: %d\n", number_of_varints);

    uint8_t debug_vals[16];

    __riscv_vse8_v_i8m1(debug_vals, data_vec, vlmax_e8m1);
    printf("data : %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
           debug_vals[0], debug_vals[1], debug_vals[2], debug_vals[3],
           debug_vals[4], debug_vals[5], debug_vals[6], debug_vals[7],
           debug_vals[8], debug_vals[9], debug_vals[10], debug_vals[11],
           debug_vals[12], debug_vals[13], debug_vals[14], debug_vals[15]);

    vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vlmax_e8m1);

    uint64_t second_bytes = createSecondBytesMask(data_vec, vlmax_e8m1);

    uint64_t compress_mask = second_bytes << 1;
    compress_mask = ~compress_mask;

    vuint64m1_t compress_mask_vec = __riscv_vmv_v_x_u64m1(compress_mask, vlmax_e64m1);
    vbool8_t comp_mask_second_bytes = __riscv_vreinterpret_v_u64m1_b8(compress_mask_vec);

    vuint64m1_t sec_vec = __riscv_vmv_v_x_u64m1(second_bytes, vlmax_e64m1);
    vbool8_t sec_mask = __riscv_vreinterpret_v_u64m1_b8(sec_vec);

    // vuint8m1_t andtest = scatter_via_gather(compressed, sec_mask, 2);

    vuint8m1_t datau08 = __riscv_vreinterpret_v_i8m1_u8m1(data_vec);

    vuint8m1_t andtest = __riscv_vmerge_vxm_u8m1(datau08, 0, __riscv_vmnot_m_b8(sec_mask, vlmax_e8m1), vlmax_e8m1);

    andtest = __riscv_vcompress_vm_u8m1(andtest, comp_mask_second_bytes, vlmax_e8m1);

    vuint32m4_t second_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(andtest, vlmax_e8m1);

    second_bytes_in_32_byte_lanes = __riscv_vand_vx_u32m4(second_bytes_in_32_byte_lanes, 0x0000007F, vlmax_e8m1 / 4);

    vbool8_t has32blane_secondbyte = __riscv_vmsne_vx_u32m4_b8(second_bytes_in_32_byte_lanes, 0, vlmax_e8m1 / 4);

    vuint8m1_t compressed_zero_msb = __riscv_vcompress_vm_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(data_vec), inverted_mask, vlmax_e8m1);
    vuint32m4_t zero_msb_bytes = __riscv_vzext_vf4_u32m4(compressed_zero_msb, vlmax_e8m1);

    zero_msb_bytes = __riscv_vsll_vx_u32m4_m(has32blane_secondbyte, zero_msb_bytes, 7, vlmax_e8m1 / 4);
    zero_msb_bytes = __riscv_vor_vv_u32m4_m(has32blane_secondbyte, zero_msb_bytes, second_bytes_in_32_byte_lanes, vlmax_e8m1 / 4);

    uint32_t result_test[16];
    __riscv_vse32_v_u32m4(result_test, zero_msb_bytes, vlmax_e8m1 / 4);
    printf("zero : %08x %d %d %08x\n",
           result_test[0], result_test[1], result_test[2], result_test[3]);

    // uint32_t result_test[16];
    __riscv_vse32_v_u32m4(result_test, second_bytes_in_32_byte_lanes, vlmax_e8m1 / 4);
    printf("sec  : %08x %08x %08x %08x\n",
           result_test[0], result_test[1], result_test[2], result_test[3]);

    // vuint8m1_t andtest = __riscv_vand_vx_u8m1_m(sec_mask, __riscv_vreinterpret_v_i8m1_u8m1(data_vec), 0x00,vlmax_e8m1);

    // __riscv_vse8_v_u8m1(debug_vals, andtest, vlmax_e8m1);
    // printf("and  : %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
    //        debug_vals[0], debug_vals[1], debug_vals[2], debug_vals[3],
    //        debug_vals[4], debug_vals[5], debug_vals[6], debug_vals[7],
    //        debug_vals[8], debug_vals[9], debug_vals[10], debug_vals[11],
    //        debug_vals[12], debug_vals[13], debug_vals[14], debug_vals[15]);

    // uint8_t *tests = (uint8_t *)second_bytes;

    // printf("result: %02x %02x %02x %02x %02x\n", tests[0], tests[1], tests[2], tests[3], tests[4]);

    // vuint8m1_t data_u8 = __riscv_vreinterpret_v_i8m1_u8m1(data_vec);

    // printf("mask: %016x\n", second_bytes);

    if (length > 0)
    {
        length = 0;
    }

    uint64_t accu = 0;

    // vuint8m1_t input_vec = __riscv_vget_v_u8m8_u8m1(varint_vec, 0);

    // uint64_t consumed = masked_vbyte_read_group(input_vec, output, mask, &ints_read, vlmax_e8m1);

    // accu += ints_read;

    // input_vec = __riscv_vslidedown_vx_u8m1(input_vec, consumed, vlmax_e8m1);
    // mask >>= consumed;
    // output += ints_read;
    // input += consumed;

    // printf("mask: %016x\n",mask);

    // consumed = masked_vbyte_read_group(input_vec, output, mask, &ints_read, vlmax_e8m1);

    // accu += ints_read;

    return accu;
}