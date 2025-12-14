#include "libvarintrvv.h"

/**
 * Just for debugging
 */
// static inline void dump_vbool8_as_bytes(const char *label, vbool8_t m, size_t vl)
// {
//     uint8_t tmp[256];

//     vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
//     vuint8m1_t ones = __riscv_vmv_v_x_u8m1(1, vl);

//     vuint8m1_t bytes = __riscv_vmerge_vvm_u8m1(zeros, ones, m, vl);

//     __riscv_vse8_v_u8m1(tmp, bytes, vl);

//     printf("%s:", label);
//     for (size_t i = 0; i < vl; i++)
//         printf(" %02x", tmp[i]);
//     printf("\n");
// }

static inline __attribute__((always_inline)) vbool8_t createSecondBytesMask(vint8m1_t data, size_t vl)
{
    vint8m1_t next_data = __riscv_vslidedown_vx_i8m1(data, 1, vl);
    vbool8_t current_set = __riscv_vmslt_vx_i8m1_b8(data, 0, vl);
    vbool8_t next_clear = __riscv_vmsge_vx_i8m1_b8(next_data, 0, vl);
    return __riscv_vmand_mm_b8(current_set, next_clear, vl);
}

static inline __attribute__((always_inline)) vbool8_t createNBytesMask(vbool8_t prev_mask, vbool8_t orig_mask, size_t vl)
{
    vuint64m1_t prev_mask_u64 = __riscv_vreinterpret_v_b8_u64m1(prev_mask);
    prev_mask_u64 = __riscv_vsrl_vx_u64m1(prev_mask_u64, 1, vl / 8);
    prev_mask_u64 = __riscv_vand_vv_u64m1(prev_mask_u64, __riscv_vreinterpret_v_b8_u64m1(orig_mask), vl / 8);
    return __riscv_vreinterpret_v_u64m1_b8(prev_mask_u64);
}

/**
 * Calculates ((upper_mask >> shift) ^ (~mask)) OR upper_mask.
 */
static inline __attribute__((always_inline)) vbool8_t createCompressMask(vbool8_t upper_mask, vbool8_t inverted_mask, size_t shifts, size_t vl)
{
    vuint64m1_t upper_mask_vec = __riscv_vreinterpret_v_b8_u64m1(upper_mask);
    vuint64m1_t shifted = __riscv_vsll_vx_u64m1(upper_mask_vec, shifts, vl / 8);

    vbool8_t msb_mask_with_third_byte_removed = __riscv_vmxor_mm_b8(__riscv_vreinterpret_v_u64m1_b8(shifted), inverted_mask, vl);

    return __riscv_vmor_mm_b8(msb_mask_with_third_byte_removed, upper_mask, vl);
}

/**
 * Number of varints that will be processed in the current loop iteration.
 */
static inline __attribute__((always_inline)) uint8_t getNumberOfVarints(vbool8_t varint_mask, size_t vl)
{
    return __riscv_vcpop_m_b8(varint_mask, vl);
}

/**
 * It's possible that the register splices a varint in half. So we can only decode complete varints in each vec.
 * This function returns the number of bytes that are occupied by complete varints in the vector register.
 */

static inline __attribute__((always_inline)) uint8_t getCompleteVarintSize(vbool8_t varint_mask, size_t vl)
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

    size_t processed = 0;

    while (length > vlmax_e8m1)
    {
        vint8m1_t data_vec = __riscv_vle8_v_i8m1((int8_t *)input, vlmax_e8m1);
        vuint8m1_t data_vec_u8 = __riscv_vreinterpret_v_i8m1_u8m1(data_vec);

        vbool8_t mask = __riscv_vmslt_vx_i8m1_b8(data_vec, 0, vlmax_e8m1); // a comparison < 0 with the data interpreted as a signed integer gives a mask with all bytes which have the MSB set

        uint8_t number_of_bytes = getCompleteVarintSize(mask, vlmax_e8m1);

        // fast path. no continuation bits set
        if (number_of_bytes == vlmax_e8m1)
        {
            __riscv_vse32_v_u32m4(output, __riscv_vzext_vf4_u32m4(data_vec_u8, vlmax_e8m1), vlmax_e8m1);
            input += number_of_bytes;
            length -= number_of_bytes;
            output += number_of_bytes;
            processed += number_of_bytes;
        }
        else
        {
            vbool8_t inverted_mask = __riscv_vmnot_m_b8(mask, vlmax_e8m1);

            uint8_t number_of_varints = getNumberOfVarints(inverted_mask, vlmax_e8m1);

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
            second_bytes_in_32_byte_lanes = __riscv_vand_vx_u32m4(second_bytes_in_32_byte_lanes, 0x0000007F, vlmax_e8m1);

            // we have to shift the result vec left by 7 to integrate the second bytes. But only the lanes actually containing second bytes
            vbool8_t has32blane_secondbyte = __riscv_vmsne_vx_u32m4_b8(second_bytes_in_32_byte_lanes, 0, vlmax_e8m1);

            result_vec = __riscv_vsll_vx_u32m4_m(has32blane_secondbyte, result_vec, 7, vlmax_e8m1);
            result_vec = __riscv_vor_vv_u32m4_m(has32blane_secondbyte, result_vec, second_bytes_in_32_byte_lanes, vlmax_e8m1);

            // any third bytes in this reg?
            vbool8_t third_bytes_mask = createNBytesMask(second_bytes_mask, mask, vlmax_e8m1);
            if (__riscv_vcpop_m_b8(third_bytes_mask, vlmax_e8m1) != 0)
            {
                vbool8_t comp_mask_third_bytes = createCompressMask(third_bytes_mask, inverted_mask, 2, vlmax_e8m1);

                vuint8m1_t zerod_msb_bytes = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(third_bytes_mask, vlmax_e8m1), vlmax_e8m1);

                vuint8m1_t compressed_third_bytes = __riscv_vcompress_vm_u8m1(zerod_msb_bytes, comp_mask_third_bytes, vlmax_e8m1);

                vuint32m4_t third_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(compressed_third_bytes, vlmax_e8m1);
                third_bytes_in_32_byte_lanes = __riscv_vand_vx_u32m4(third_bytes_in_32_byte_lanes, 0x0000007F, vlmax_e8m1);

                vbool8_t has32blane_thirdbyte = __riscv_vmsne_vx_u32m4_b8(third_bytes_in_32_byte_lanes, 0, vlmax_e8m1);

                result_vec = __riscv_vsll_vx_u32m4_m(has32blane_thirdbyte, result_vec, 7, vlmax_e8m1);

                result_vec = __riscv_vor_vv_u32m4_m(has32blane_thirdbyte, result_vec, third_bytes_in_32_byte_lanes, vlmax_e8m1);

                vbool8_t fourth_bytes_mask = createNBytesMask(third_bytes_mask, mask, vlmax_e8m1);
                if (__riscv_vcpop_m_b8(fourth_bytes_mask, vlmax_e8m1) != 0)
                {
                    vbool8_t comp_mask_fourth_bytes = createCompressMask(fourth_bytes_mask, inverted_mask, 3, vlmax_e8m1);

                    zerod_msb_bytes = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(fourth_bytes_mask, vlmax_e8m1), vlmax_e8m1);

                    vuint8m1_t compressed_fourth_bytes = __riscv_vcompress_vm_u8m1(zerod_msb_bytes, comp_mask_fourth_bytes, vlmax_e8m1);

                    vuint32m4_t fourth_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(compressed_fourth_bytes, vlmax_e8m1);
                    fourth_bytes_in_32_byte_lanes = __riscv_vand_vx_u32m4(fourth_bytes_in_32_byte_lanes, 0x0000007F, vlmax_e8m1);

                    vbool8_t has32blane_fourthbyte = __riscv_vmsne_vx_u32m4_b8(fourth_bytes_in_32_byte_lanes, 0, vlmax_e8m1);

                    result_vec = __riscv_vsll_vx_u32m4_m(has32blane_fourthbyte, result_vec, 7, vlmax_e8m1);

                    result_vec = __riscv_vor_vv_u32m4_m(has32blane_fourthbyte, result_vec, fourth_bytes_in_32_byte_lanes, vlmax_e8m1);

                    vbool8_t fifth_bytes_mask = createNBytesMask(fourth_bytes_mask, mask, vlmax_e8m1);
                    if (__riscv_vcpop_m_b8(fifth_bytes_mask, vlmax_e8m1) != 0)
                    {
                        vbool8_t comp_mask_fifth_bytes = createCompressMask(fifth_bytes_mask, inverted_mask, 4, vlmax_e8m1);

                        zerod_msb_bytes = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(fifth_bytes_mask, vlmax_e8m1), vlmax_e8m1);

                        vuint8m1_t compressed_fifth_bytes = __riscv_vcompress_vm_u8m1(zerod_msb_bytes, comp_mask_fifth_bytes, vlmax_e8m1);

                        vuint32m4_t fifth_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(compressed_fifth_bytes, vlmax_e8m1);
                        fifth_bytes_in_32_byte_lanes = __riscv_vand_vx_u32m4(fifth_bytes_in_32_byte_lanes, 0x0000007F, vlmax_e8m1);

                        vbool8_t has32blane_fifthbyte = __riscv_vmsne_vx_u32m4_b8(fifth_bytes_in_32_byte_lanes, 0, vlmax_e8m1);

                        result_vec = __riscv_vsll_vx_u32m4_m(has32blane_fifthbyte, result_vec, 7, vlmax_e8m1);

                        result_vec = __riscv_vor_vv_u32m4_m(has32blane_fifthbyte, result_vec, fifth_bytes_in_32_byte_lanes, vlmax_e8m1);
                    }
                }
            }
            // only save the actual number of complete varints to memory
            __riscv_vse32_v_u32m4(output, result_vec, number_of_varints);

            // loop bookkeeping
            input += number_of_bytes;
            length -= number_of_bytes;
            output += number_of_varints;

            processed += number_of_varints;
        }
    }

    return processed;
}