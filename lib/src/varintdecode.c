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

static inline void dump_vbool8_as_bytes(const char *label, vbool8_t m, size_t vl)
{
    uint8_t tmp[256]; // or malloc; must be >= vl bytes

    vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
    vuint8m1_t ones = __riscv_vmv_v_x_u8m1(1, vl);

    vuint8m1_t bytes = __riscv_vmerge_vvm_u8m1(zeros, ones, m, vl);

    __riscv_vse8_v_u8m1(tmp, bytes, vl);

    printf("%s:", label);
    for (size_t i = 0; i < vl; i++)
        printf(" %02x", tmp[i]);
    printf("\n");
}

static inline vbool8_t createSecondBytesMask(vint8m1_t data, size_t vl)
{

    // vbool8_t mask = __riscv_vmslt_vx_i8m1_b8(data, 0, vl);                                             // a comparison < 0 with the data interpreted as a signed integer gives us a mask with all bytes which have the MSB set
    // vuint64m1_t mask_as_u64 = __riscv_vreinterpret_v_u8m1_u64m1(__riscv_vreinterpret_v_b8_u8m1(mask)); // Type conversion: b8 -> u8 -> u64
    // __riscv_vsll_vx_u64m1(mask_as_u64,1)

    vint8m1_t next_data = __riscv_vslidedown_vx_i8m1(data, 1, vl);
    vbool8_t current_set = __riscv_vmslt_vx_i8m1_b8(data, 0, vl);
    vbool8_t next_clear = __riscv_vmsge_vx_i8m1_b8(next_data, 0, vl);
    return __riscv_vmand_mm_b8(current_set, next_clear, vl);

    // uint64_t masku8 = createMask(data, vl);
    // uint64_t mask_shifted = masku8 << 1;
    // mask_shifted = ~mask_shifted;
    // return masku8 & mask_shifted;
}

static inline vbool8_t createNBytesMask(vbool8_t prev_mask, vbool8_t orig_mask, size_t vl)
{
    vuint64m1_t prev_mask_u64 = __riscv_vreinterpret_v_b8_u64m1(prev_mask);
    prev_mask_u64 = __riscv_vsrl_vx_u64m1(prev_mask_u64, 1, vl / 8);

    // dump_vbool8_as_bytes("maski", __riscv_vreinterpret_v_u64m1_b8(prev_mask_u64), vl);
    prev_mask_u64 = __riscv_vand_vv_u64m1(prev_mask_u64, __riscv_vreinterpret_v_b8_u64m1(orig_mask), vl / 8);
    return __riscv_vreinterpret_v_u64m1_b8(prev_mask_u64);
}

/**
 * Calculates ((upper_mask >> shift) ^ (~mask)) OR upper_mask.
 */
static inline vbool8_t createCompressMask(vbool8_t upper_mask, vbool8_t inverted_mask, size_t shifts, size_t vl)
{
    vuint64m1_t upper_mask_vec = __riscv_vreinterpret_v_b8_u64m1(upper_mask);
    vuint64m1_t shifted = __riscv_vsll_vx_u64m1(upper_mask_vec, shifts, vl / 8);

    vbool8_t msb_mask_with_third_byte_removed = __riscv_vmxor_mm_b8(__riscv_vreinterpret_v_u64m1_b8(shifted), inverted_mask, vl);

    return __riscv_vmor_mm_b8(msb_mask_with_third_byte_removed, upper_mask, vl);
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
static inline uint8_t getCompleteVarintSize(vbool8_t varint_mask, size_t vl)
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

/**
 * input: uint8_t pointer to the start of the compressed varints
 * output: decompressed 32-bit integers
 * length: size of the varints in bytes
 * returns: uint64_t number of decompressed integers
 */
uint64_t varint_decode(uint8_t *input, uint32_t *output, size_t length)
{
    const size_t vlmax_e8m1 = __riscv_vsetvlmax_e8m1();

    vint8m1_t data_vec = __riscv_vle8_v_i8m1((int8_t *)input, vlmax_e8m1);
    vuint8m1_t data_vec_u8 = __riscv_vreinterpret_v_i8m1_u8m1(data_vec);

    vbool8_t mask = __riscv_vmslt_vx_i8m1_b8(data_vec, 0, vlmax_e8m1); // a comparison < 0 with the data interpreted as a signed integer gives a mask with all bytes which have the MSB set

    vbool8_t inverted_mask = __riscv_vmnot_m_b8(mask, vlmax_e8m1);

    uint8_t number_of_bytes = getCompleteVarintSize(mask, vlmax_e8m1);

    printf("Number of Bytes: %d\n", number_of_bytes);

    uint8_t number_of_varints = getNumberOfVarints(inverted_mask, vlmax_e8m1);

    printf("Number of Varints: %d\n", number_of_varints);

    // first step: move all bytes with MSB==0 in their own 32-bit lane
    vuint8m1_t compressed_zero_msb = __riscv_vcompress_vm_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(data_vec), inverted_mask, vlmax_e8m1);
    vuint32m4_t result_vec = __riscv_vzext_vf4_u32m4(compressed_zero_msb, vlmax_e8m1);

    // create mask which selects all second bytes of multi-byte varints
    vbool8_t second_bytes_mask = createSecondBytesMask(data_vec, vlmax_e8m1);

    // null all lanes which are not second byte lanes
    vuint8m1_t multi_byte_varint_vec = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(second_bytes_mask, vlmax_e8m1), vlmax_e8m1);

    vbool8_t comp_mask_second_bytes = createCompressMask(second_bytes_mask, inverted_mask, 1, vlmax_e8m1);

    // compress second bytes so they line up with the 32-bit lanes their corresponding first bytes
    multi_byte_varint_vec = __riscv_vcompress_vm_u8m1(multi_byte_varint_vec, comp_mask_second_bytes, vlmax_e8m1);

    vuint32m4_t second_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(multi_byte_varint_vec, vlmax_e8m1);

    // mask of continuation bit
    second_bytes_in_32_byte_lanes = __riscv_vand_vx_u32m4(second_bytes_in_32_byte_lanes, 0x0000007F, vlmax_e8m1 / 4);

    // we have to shift the result vec left by 7 to integrate the second bytes. But only the lanes actually containing second bytes
    vbool8_t has32blane_secondbyte = __riscv_vmsne_vx_u32m4_b8(second_bytes_in_32_byte_lanes, 0, vlmax_e8m1 / 4);

    result_vec = __riscv_vsll_vx_u32m4_m(has32blane_secondbyte, result_vec, 7, vlmax_e8m1 / 4);
    result_vec = __riscv_vor_vv_u32m4_m(has32blane_secondbyte, result_vec, second_bytes_in_32_byte_lanes, vlmax_e8m1 / 4);

    // any third bytes in this reg?
    vbool8_t third_bytes_mask = createNBytesMask(second_bytes_mask, mask, vlmax_e8m1);
    if (__riscv_vcpop_m_b8(third_bytes_mask, vlmax_e8m1) != 0)
    {
        vbool8_t comp_mask_third_bytes = createCompressMask(third_bytes_mask, inverted_mask, 2, vlmax_e8m1);

        vuint8m1_t zerod_msb_bytes = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(third_bytes_mask, vlmax_e8m1), vlmax_e8m1);

        vuint8m1_t compressed_third_bytes = __riscv_vcompress_vm_u8m1(zerod_msb_bytes, comp_mask_third_bytes, vlmax_e8m1);

        vuint32m4_t third_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(compressed_third_bytes, vlmax_e8m1);
        third_bytes_in_32_byte_lanes = __riscv_vand_vx_u32m4(third_bytes_in_32_byte_lanes, 0x0000007F, vlmax_e8m1 / 4);

        vbool8_t has32blane_thirdbyte = __riscv_vmsne_vx_u32m4_b8(third_bytes_in_32_byte_lanes, 0, vlmax_e8m1 / 4);

        result_vec = __riscv_vsll_vx_u32m4_m(has32blane_thirdbyte, result_vec, 7, vlmax_e8m1 / 4);

        result_vec = __riscv_vor_vv_u32m4_m(has32blane_thirdbyte, result_vec, third_bytes_in_32_byte_lanes, vlmax_e8m1 / 4);

        vbool8_t fourth_bytes_mask = createNBytesMask(third_bytes_mask, mask, vlmax_e8m1);
        if (__riscv_vcpop_m_b8(fourth_bytes_mask, vlmax_e8m1) != 0)
        {
            vbool8_t comp_mask_fourth_bytes = createCompressMask(fourth_bytes_mask, inverted_mask, 3, vlmax_e8m1);

            zerod_msb_bytes = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(fourth_bytes_mask, vlmax_e8m1), vlmax_e8m1);

            vuint8m1_t compressed_fourth_bytes = __riscv_vcompress_vm_u8m1(zerod_msb_bytes, comp_mask_fourth_bytes, vlmax_e8m1);

            vuint32m4_t fourth_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(compressed_fourth_bytes, vlmax_e8m1);
            fourth_bytes_in_32_byte_lanes = __riscv_vand_vx_u32m4(fourth_bytes_in_32_byte_lanes, 0x0000007F, vlmax_e8m1 / 4);

            vbool8_t has32blane_fourthbyte = __riscv_vmsne_vx_u32m4_b8(fourth_bytes_in_32_byte_lanes, 0, vlmax_e8m1 / 4);

            result_vec = __riscv_vsll_vx_u32m4_m(has32blane_fourthbyte, result_vec, 7, vlmax_e8m1 / 4);

            result_vec = __riscv_vor_vv_u32m4_m(has32blane_fourthbyte, result_vec, fourth_bytes_in_32_byte_lanes, vlmax_e8m1 / 4);

            vbool8_t fifth_bytes_mask = createNBytesMask(fourth_bytes_mask, mask, vlmax_e8m1);
            if (__riscv_vcpop_m_b8(fifth_bytes_mask, vlmax_e8m1) != 0)
            {
                vbool8_t comp_mask_fifth_bytes = createCompressMask(fifth_bytes_mask, inverted_mask, 4, vlmax_e8m1);

                zerod_msb_bytes = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(fifth_bytes_mask, vlmax_e8m1), vlmax_e8m1);

                vuint8m1_t compressed_fifth_bytes = __riscv_vcompress_vm_u8m1(zerod_msb_bytes, comp_mask_fifth_bytes, vlmax_e8m1);

                vuint32m4_t fifth_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(compressed_fifth_bytes, vlmax_e8m1);
                fifth_bytes_in_32_byte_lanes = __riscv_vand_vx_u32m4(fifth_bytes_in_32_byte_lanes, 0x0000007F, vlmax_e8m1 / 4);

                vbool8_t has32blane_fifthbyte = __riscv_vmsne_vx_u32m4_b8(fifth_bytes_in_32_byte_lanes, 0, vlmax_e8m1 / 4);

                result_vec = __riscv_vsll_vx_u32m4_m(has32blane_fifthbyte, result_vec, 7, vlmax_e8m1 / 4);

                result_vec = __riscv_vor_vv_u32m4_m(has32blane_fifthbyte, result_vec, fifth_bytes_in_32_byte_lanes, vlmax_e8m1 / 4);
            }
        }

        __riscv_vse32_v_u32m4(output, result_vec, vlmax_e8m1 / 4);
    }

    if (length > 0)
    {
        length = 0;
    }
    return number_of_varints;
}