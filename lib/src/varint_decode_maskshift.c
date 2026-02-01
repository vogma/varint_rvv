#include "libvarintrvv.h"

static inline __attribute__((always_inline)) vbool8_t createSecondBytesMask(vint8m1_t data, size_t vl)
{
    vint8m1_t next_data = __riscv_vslidedown_vx_i8m1(data, 1, vl);
    vbool8_t current_set = __riscv_vmslt_vx_i8m1_b8(data, 0, vl);
    vbool8_t next_clear = __riscv_vmsge_vx_i8m1_b8(next_data, 0, vl);
    return __riscv_vmand_mm_b8(current_set, next_clear, vl);
}

static inline __attribute__((always_inline)) vbool4_t createSecondBytesMask_m2(vint8m2_t data, size_t vl)
{
    vint8m2_t next_data = __riscv_vslidedown_vx_i8m2(data, 1, vl);
    vbool4_t current_set = __riscv_vmslt_vx_i8m2_b4(data, 0, vl);
    vbool4_t next_clear = __riscv_vmsge_vx_i8m2_b4(next_data, 0, vl);
    return __riscv_vmand_mm_b4(current_set, next_clear, vl);
}

static inline __attribute__((always_inline)) vbool8_t createNBytesMask(vbool8_t prev_mask, vbool8_t orig_mask, size_t vl)
{
    vuint64m1_t prev_mask_u64 = __riscv_vreinterpret_v_b8_u64m1(prev_mask);
    prev_mask_u64 = __riscv_vsrl_vx_u64m1(prev_mask_u64, 1, vl / 8);
    prev_mask_u64 = __riscv_vand_vv_u64m1(prev_mask_u64, __riscv_vreinterpret_v_b8_u64m1(orig_mask), vl / 8);
    return __riscv_vreinterpret_v_u64m1_b8(prev_mask_u64);
    // return prev_mask;
}

static inline __attribute__((always_inline)) vbool4_t createNBytesMask_m2(vbool4_t prev_mask, vbool4_t orig_mask, size_t vl)
{
    vuint64m1_t prev_mask_u64 = __riscv_vreinterpret_v_u32m1_u64m1(__riscv_vreinterpret_v_b4_u32m1(prev_mask));
    prev_mask_u64 = __riscv_vsrl_vx_u64m1(prev_mask_u64, 1, vl / 8);
    prev_mask_u64 = __riscv_vand_vv_u64m1(prev_mask_u64, __riscv_vreinterpret_v_u32m1_u64m1(__riscv_vreinterpret_v_b4_u32m1(orig_mask)), vl / 8);
    return __riscv_vreinterpret_v_u8m1_b4(__riscv_vreinterpret_v_u64m1_u8m1(prev_mask_u64));
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
    // return upper_mask;
}

/**
 * Calculates ((upper_mask >> shift) ^ (~mask)) OR upper_mask.
 */
static inline __attribute__((always_inline)) vbool4_t createCompressMask_m2(vbool4_t upper_mask, vbool4_t inverted_mask, size_t shifts, size_t vl)
{
    vuint64m1_t upper_mask_vec = __riscv_vreinterpret_v_u32m1_u64m1(__riscv_vreinterpret_v_b4_u32m1(upper_mask));
    vuint64m1_t shifted = __riscv_vsll_vx_u64m1(upper_mask_vec, shifts, vl / 8);

    vbool4_t msb_mask_with_third_byte_removed = __riscv_vmxor_mm_b4(__riscv_vreinterpret_v_u8m1_b4(__riscv_vreinterpret_v_u64m1_u8m1(shifted)), inverted_mask, vl);

    return __riscv_vmor_mm_b4(msb_mask_with_third_byte_removed, upper_mask, vl);
}

/**
 * Number of varints that will be processed in the current loop iteration.
 */
static inline __attribute__((always_inline)) uint8_t getNumberOfVarints(vbool8_t varint_mask, size_t vl)
{
    return __riscv_vcpop_m_b8(varint_mask, vl);
}

/**
 * Number of varints that will be processed in the current loop iteration.
 */
static inline __attribute__((always_inline)) uint8_t getNumberOfVarints_m2(vbool4_t varint_mask, size_t vl)
{
    return __riscv_vcpop_m_b4(varint_mask, vl);
}

/**
 * It's possible that the register splices a varint in half. So we can only decode complete varints in each vec.
 * This function returns the number of bytes that are occupied by complete varints in the vector register.
 */

static inline __attribute__((always_inline)) uint8_t getCompleteVarintSize_m2(vbool4_t varint_mask, size_t vl)
{
    // every lane gets an index
    vuint8m2_t index_vec = __riscv_vid_v_u8m2(vl);

    // set every byte which has the continuation bit set to zero
    vuint8m2_t index_of_varints = __riscv_vmerge_vxm_u8m2(index_vec, 0, varint_mask, vl);

    // find the largest index value and place in the lowest lane
    vint8m1_t resultindex = __riscv_vredmax_vs_i8m2_i8m1(__riscv_vreinterpret_v_u8m2_i8m2(index_of_varints), __riscv_vmv_v_x_i8m1(0, vl), vl);

    // return lowest lane in scalar reg (+1 to convert Index (0-based) to Count (1-based))
    return __riscv_vmv_x_s_i8m1_i8(resultindex) + 1;
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
size_t varint_decode_m2(const uint8_t *input, size_t length, uint32_t *output)
{
    size_t processed = 0;

    size_t vl;

    while (length > 0)
    {
        vl = __riscv_vsetvl_e8m2(length);

        vint8m2_t data_vec = __riscv_vle8_v_i8m2((int8_t *)input, vl);
        vuint8m2_t data_vec_u8 = __riscv_vreinterpret_v_i8m2_u8m2(data_vec);

        vbool4_t inverted_mask = __riscv_vmsge_vx_i8m2_b4(data_vec, 0, vl);
        uint8_t number_of_varints = getNumberOfVarints_m2(inverted_mask, vl);

        // fast path. no continuation bits set
        if (__builtin_expect(number_of_varints == vl , 1))
        {
            // expand every byte to 32-bit lane and save to memory
            __riscv_vse32_v_u32m8(output, __riscv_vzext_vf4_u32m8(data_vec_u8, vl), vl);
            input += number_of_varints;
            length -= number_of_varints;
            output += number_of_varints;
            processed += number_of_varints;
        }
        else
        {
            vbool4_t mask = __riscv_vmnot_m_b4(inverted_mask, vl);

            uint8_t number_of_bytes = getCompleteVarintSize_m2(mask, vl);

            // first step: move all bytes with MSB==0 in their own 32-bit lane
            vuint8m2_t compressed_zero_msb = __riscv_vcompress_vm_u8m2(__riscv_vreinterpret_v_i8m2_u8m2(data_vec), inverted_mask, vl);
            vuint32m8_t result_vec = __riscv_vzext_vf4_u32m8(compressed_zero_msb, vl);

            // create mask which selects all second bytes of multi-byte varints
            vbool4_t second_bytes_mask = createSecondBytesMask_m2(data_vec, vl);

            // null out all lanes which are not second byte lanes
            vuint8m2_t multi_byte_varint_vec = __riscv_vmerge_vxm_u8m2(data_vec_u8, 0, __riscv_vmnot_m_b4(second_bytes_mask, vl), vl);

            vbool4_t comp_mask_second_bytes = createCompressMask_m2(second_bytes_mask, inverted_mask, 1, vl);

            // compress second bytes so they line up with the 32-bit lanes their corresponding first bytes
            multi_byte_varint_vec = __riscv_vcompress_vm_u8m2(multi_byte_varint_vec, comp_mask_second_bytes, vl);

            vbool4_t has_lane_secondbyte = __riscv_vmsne_vx_u8m2_b4(multi_byte_varint_vec, 0, vl);

            // remove continuation bit
            multi_byte_varint_vec = __riscv_vand_vx_u8m2(multi_byte_varint_vec, 0x7F, vl);

            // expand to 32-bit lanes
            vuint32m8_t second_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m8(multi_byte_varint_vec, vl);

            // we have to shift the result vec left by 7 to integrate the second bytes. But only the lanes actually containing second bytes
            result_vec = __riscv_vsll_vx_u32m8_m(has_lane_secondbyte, result_vec, 7, vl);
            result_vec = __riscv_vor_vv_u32m8_m(has_lane_secondbyte, result_vec, second_bytes_in_32_byte_lanes, vl);

            // any third bytes in this reg?
            vbool4_t third_bytes_mask = createNBytesMask_m2(second_bytes_mask, mask, vl);
            if (__riscv_vcpop_m_b4(third_bytes_mask, vl) != 0)
            {
                vbool4_t comp_mask_third_bytes = createCompressMask_m2(third_bytes_mask, inverted_mask, 2, vl);
                vuint8m2_t zerod_msb_bytes = __riscv_vmerge_vxm_u8m2(data_vec_u8, 0, __riscv_vmnot_m_b4(third_bytes_mask, vl), vl);
                vuint8m2_t compressed_third_bytes = __riscv_vcompress_vm_u8m2(zerod_msb_bytes, comp_mask_third_bytes, vl);
                vbool4_t has32blane_thirdbyte = __riscv_vmsne_vx_u8m2_b4(compressed_third_bytes, 0, vl);
                compressed_third_bytes = __riscv_vand_vx_u8m2(compressed_third_bytes, 0x7F, vl);
                vuint32m8_t third_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m8(compressed_third_bytes, vl);
                result_vec = __riscv_vsll_vx_u32m8_m(has32blane_thirdbyte, result_vec, 7, vl);
                result_vec = __riscv_vor_vv_u32m8_m(has32blane_thirdbyte, result_vec, third_bytes_in_32_byte_lanes, vl);

                vbool4_t fourth_bytes_mask = createNBytesMask_m2(third_bytes_mask, mask, vl);
                if (__riscv_vcpop_m_b4(fourth_bytes_mask, vl) != 0)
                {
                    vbool4_t comp_mask_fourth_bytes = createCompressMask_m2(fourth_bytes_mask, inverted_mask, 3, vl);
                    zerod_msb_bytes = __riscv_vmerge_vxm_u8m2(data_vec_u8, 0, __riscv_vmnot_m_b4(fourth_bytes_mask, vl), vl);
                    vuint8m2_t compressed_fourth_bytes = __riscv_vcompress_vm_u8m2(zerod_msb_bytes, comp_mask_fourth_bytes, vl);
                    vbool4_t has32blane_fourthbyte = __riscv_vmsne_vx_u8m2_b4(compressed_fourth_bytes, 0, vl);
                    compressed_fourth_bytes = __riscv_vand_vx_u8m2(compressed_fourth_bytes, 0x0000007F, vl);
                    vuint32m8_t fourth_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m8(compressed_fourth_bytes, vl);
                    result_vec = __riscv_vsll_vx_u32m8_m(has32blane_fourthbyte, result_vec, 7, vl);
                    result_vec = __riscv_vor_vv_u32m8_m(has32blane_fourthbyte, result_vec, fourth_bytes_in_32_byte_lanes, vl);

                    vbool4_t fifth_bytes_mask = createNBytesMask_m2(fourth_bytes_mask, mask, vl);
                    if (__riscv_vcpop_m_b4(fifth_bytes_mask, vl) != 0)
                    {
                        vbool4_t comp_mask_fifth_bytes = createCompressMask_m2(fifth_bytes_mask, inverted_mask, 4, vl);
                        zerod_msb_bytes = __riscv_vmerge_vxm_u8m2(data_vec_u8, 0, __riscv_vmnot_m_b4(fifth_bytes_mask, vl), vl);
                        vuint8m2_t compressed_fifth_bytes = __riscv_vcompress_vm_u8m2(zerod_msb_bytes, comp_mask_fifth_bytes, vl);
                        vbool4_t has32blane_fifthbyte = __riscv_vmsne_vx_u8m2_b4(compressed_fifth_bytes, 0, vl);
                        compressed_fifth_bytes = __riscv_vand_vx_u8m2(compressed_fifth_bytes, 0x7F, vl);
                        vuint32m8_t fifth_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m8(compressed_fifth_bytes, vl);
                        result_vec = __riscv_vsll_vx_u32m8_m(has32blane_fifthbyte, result_vec, 7, vl);
                        result_vec = __riscv_vor_vv_u32m8_m(has32blane_fifthbyte, result_vec, fifth_bytes_in_32_byte_lanes, vl);
                    }
                }
            }
            // only save the actual number of complete varints to memory
            __riscv_vse32_v_u32m8(output, result_vec, number_of_varints);

            // loop bookkeeping
            input += number_of_bytes;
            length -= number_of_bytes;
            output += number_of_varints;
            processed += number_of_varints;
        }
    }
    return processed;
}

/**
 * input: uint8_t pointer to the start of the compressed varints
 * output: decompressed 32-bit integers
 * length: size of the varints in bytes
 * returns: uint64_t number of decompressed integers
 */
size_t varint_decode_m1(const uint8_t *input, size_t length, uint32_t *output)
{
    size_t processed = 0;

    size_t vl;

    while (length > 0)
    {
        vl = __riscv_vsetvl_e8m1(length);

        vuint8m1_t data_vec_u8 = __riscv_vle8_v_u8m1((int8_t *)input, vl);
        // vuint8m1_t data_vec_u8 = __riscv_vreinterpret_v_i8m1_u8m1(data_vec);

        vbool8_t inverted_mask = __riscv_vmsge_vx_i8m1_b8(__riscv_vreinterpret_v_u8m1_i8m1(data_vec_u8), 0, vl);

        uint8_t number_of_varints = getNumberOfVarints(inverted_mask, vl);

        // fast path. no continuation bits set (hint: expect this to be true)
        if (__builtin_expect((number_of_varints == vl), 1))
        // if (number_of_varints == vl)
        {
            // expand every byte to 32-bit lane and save to memory
            __riscv_vse32_v_u32m4(output, __riscv_vzext_vf4_u32m4(data_vec_u8, vl), vl);
            input += vl;
            length -= vl;
            output += vl;
            processed += vl;
        }
        else
        {

            vint8m1_t data_vec = __riscv_vreinterpret_v_u8m1_i8m1(data_vec_u8);
            vbool8_t mask = __riscv_vmnot_m_b8(inverted_mask, vl);

            uint8_t number_of_bytes = getCompleteVarintSize(mask, vl);

            // first step: move all bytes with MSB==0 in their own 32-bit lane
            vuint8m1_t compressed_zero_msb = __riscv_vcompress_vm_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(data_vec), inverted_mask, vl);
            vuint32m4_t result_vec = __riscv_vzext_vf4_u32m4(compressed_zero_msb, vl);

            // create mask which selects all second bytes of multi-byte varints
            vbool8_t second_bytes_mask = createSecondBytesMask(data_vec, vl);

            // null out all lanes which are not second byte lanes
            vuint8m1_t multi_byte_varint_vec = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(second_bytes_mask, vl), vl);

            vbool8_t comp_mask_second_bytes = createCompressMask(second_bytes_mask, inverted_mask, 1, vl);

            // compress second bytes so they line up with the 32-bit lanes their corresponding first bytes
            multi_byte_varint_vec = __riscv_vcompress_vm_u8m1(multi_byte_varint_vec, comp_mask_second_bytes, vl);

            vbool8_t has_lane_secondbyte = __riscv_vmsne_vx_u8m1_b8(multi_byte_varint_vec, 0, vl);

            // mask of continuation bit
            multi_byte_varint_vec = __riscv_vand_vx_u8m1(multi_byte_varint_vec, 0x7F, vl);

            // expand to 32-bit lanes
            vuint32m4_t second_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(multi_byte_varint_vec, vl);

            // we have to shift the result vec left by 7 to integrate the second bytes. But only the lanes actually containing second bytes
            result_vec = __riscv_vsll_vx_u32m4_m(has_lane_secondbyte, result_vec, 7, vl);
            result_vec = __riscv_vor_vv_u32m4_m(has_lane_secondbyte, result_vec, second_bytes_in_32_byte_lanes, vl);

            // any third bytes in this reg?
            vbool8_t third_bytes_mask = createNBytesMask(second_bytes_mask, mask, vl);
            if (__riscv_vcpop_m_b8(third_bytes_mask, vl) != 0)
            {
                vbool8_t comp_mask_third_bytes = createCompressMask(third_bytes_mask, inverted_mask, 2, vl);
                vuint8m1_t zerod_msb_bytes = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(third_bytes_mask, vl), vl);
                vuint8m1_t compressed_third_bytes = __riscv_vcompress_vm_u8m1(zerod_msb_bytes, comp_mask_third_bytes, vl);
                vbool8_t has32blane_thirdbyte = __riscv_vmsne_vx_u8m1_b8(compressed_third_bytes, 0, vl);
                compressed_third_bytes = __riscv_vand_vx_u8m1(compressed_third_bytes, 0x7F, vl);
                vuint32m4_t third_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(compressed_third_bytes, vl);
                result_vec = __riscv_vsll_vx_u32m4_m(has32blane_thirdbyte, result_vec, 7, vl);
                result_vec = __riscv_vor_vv_u32m4_m(has32blane_thirdbyte, result_vec, third_bytes_in_32_byte_lanes, vl);

                vbool8_t fourth_bytes_mask = createNBytesMask(third_bytes_mask, mask, vl);
                if (__riscv_vcpop_m_b8(fourth_bytes_mask, vl) != 0)
                {
                    vbool8_t comp_mask_fourth_bytes = createCompressMask(fourth_bytes_mask, inverted_mask, 3, vl);
                    zerod_msb_bytes = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(fourth_bytes_mask, vl), vl);
                    vuint8m1_t compressed_fourth_bytes = __riscv_vcompress_vm_u8m1(zerod_msb_bytes, comp_mask_fourth_bytes, vl);
                    vbool8_t has32blane_fourthbyte = __riscv_vmsne_vx_u8m1_b8(compressed_fourth_bytes, 0, vl);
                    compressed_fourth_bytes = __riscv_vand_vx_u8m1(compressed_fourth_bytes, 0x0000007F, vl);
                    vuint32m4_t fourth_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(compressed_fourth_bytes, vl);
                    result_vec = __riscv_vsll_vx_u32m4_m(has32blane_fourthbyte, result_vec, 7, vl);
                    result_vec = __riscv_vor_vv_u32m4_m(has32blane_fourthbyte, result_vec, fourth_bytes_in_32_byte_lanes, vl);

                    vbool8_t fifth_bytes_mask = createNBytesMask(fourth_bytes_mask, mask, vl);
                    if (__riscv_vcpop_m_b8(fifth_bytes_mask, vl) != 0)
                    {
                        vbool8_t comp_mask_fifth_bytes = createCompressMask(fifth_bytes_mask, inverted_mask, 4, vl);
                        zerod_msb_bytes = __riscv_vmerge_vxm_u8m1(data_vec_u8, 0, __riscv_vmnot_m_b8(fifth_bytes_mask, vl), vl);
                        vuint8m1_t compressed_fifth_bytes = __riscv_vcompress_vm_u8m1(zerod_msb_bytes, comp_mask_fifth_bytes, vl);
                        vbool8_t has32blane_fifthbyte = __riscv_vmsne_vx_u8m1_b8(compressed_fifth_bytes, 0, vl);
                        compressed_fifth_bytes = __riscv_vand_vx_u8m1(compressed_fifth_bytes, 0x7F, vl);
                        vuint32m4_t fifth_bytes_in_32_byte_lanes = __riscv_vzext_vf4_u32m4(compressed_fifth_bytes, vl);
                        result_vec = __riscv_vsll_vx_u32m4_m(has32blane_fifthbyte, result_vec, 7, vl);
                        result_vec = __riscv_vor_vv_u32m4_m(has32blane_fifthbyte, result_vec, fifth_bytes_in_32_byte_lanes, vl);
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

size_t varint_decode(const uint8_t *input, size_t length, uint32_t *output)
{
    const size_t vlmax_e8m1 = __riscv_vsetvlmax_e8m1();

    if (vlmax_e8m1 < 64)
    {
        return varint_decode_m2(input, length, output);
    }
    else
    {
        return varint_decode_m1(input, length, output);
    }
}