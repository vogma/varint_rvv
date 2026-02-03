#include "libvarintrvv.h"
#include <stdio.h>

/**
 * Number of varints that will be processed in the current loop iteration.
 */
static inline __attribute__((always_inline)) uint8_t getNumberOfVarints(vbool8_t varint_mask, size_t vl)
{
    return __riscv_vcpop_m_b8(varint_mask, vl);
}

/**
 * input: uint8_t pointer to the start of the compressed varints
 * output: decompressed 32-bit integers
 * length: size of the varints in bytes
 * returns: size_t number of decompressed integers
 */
size_t varint_decode_vecshift(const uint8_t *input, size_t length, uint32_t *output)
{
    size_t processed = 0;

    size_t vl;

    while (length > 0)
    {
        vl = __riscv_vsetvl_e8m1(length);

        vuint8m1_t data_vec_u8 = __riscv_vle8_v_u8m1((int8_t *)input, vl);

        vbool8_t termination_mask = __riscv_vmsge_vx_i8m1_b8(__riscv_vreinterpret_v_u8m1_i8m1(data_vec_u8), 0, vl);

        uint8_t number_of_varints = getNumberOfVarints(termination_mask, vl);

        // fast path. no continuation bits set
        if (number_of_varints == vl)
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

            // inspired by https://github.com/camel-cdr/rvv-bench/blob/main/vector-utf/8toN_gather.c
            vuint8m1_t v1 = __riscv_vslide1down_vx_u8m1(data_vec_u8, 0, vl);
            vuint8m1_t v2 = __riscv_vslide1down_vx_u8m1(v1, 0, vl);
            vuint8m1_t v3 = __riscv_vslide1down_vx_u8m1(v2, 0, vl);
            vuint8m1_t v4 = __riscv_vslide1down_vx_u8m1(v3, 0, vl);

            // every byte after a termination byte is as first byte
            vuint8m1_t v_prev = __riscv_vslide1up(data_vec_u8, 0, vl);
            vbool8_t m_first_bytes = __riscv_vmsleu_vx_u8m1_b8(v_prev, 0x7F, vl);

            // compress the slided input data vectors with the first_byte mask. That way we get the second, third, fourth and fifth byte after each first byte.
            vuint8m1_t first_bytes = __riscv_vcompress_vm_u8m1(data_vec_u8, m_first_bytes, vl);
            vuint8m1_t second_bytes = __riscv_vcompress_vm_u8m1(v1, m_first_bytes, vl);
            vuint8m1_t third_bytes = __riscv_vcompress_vm_u8m1(v2, m_first_bytes, vl);
            vuint8m1_t fourth_bytes = __riscv_vcompress_vm_u8m1(v3, m_first_bytes, vl);
            vuint8m1_t fifth_bytes = __riscv_vcompress_vm_u8m1(v4, m_first_bytes, vl);

            vbool8_t m_second_bytes = __riscv_vmsgtu_vx_u8m1_b8(first_bytes, 0x7F, vl);
            vbool8_t m_third_bytes = __riscv_vmand_mm_b8(m_second_bytes, __riscv_vmsgtu_vx_u8m1_b8(second_bytes, 0x7F, vl), vl);
            vbool8_t m_fourth_bytes = __riscv_vmand_mm_b8(m_third_bytes, __riscv_vmsgtu_vx_u8m1_b8(third_bytes, 0x7F, vl), vl);
            vbool8_t m_fifth_bytes = __riscv_vmand_mm_b8(m_fourth_bytes, __riscv_vmsgtu_vx_u8m1_b8(fourth_bytes, 0x7F, vl), vl);

            // Compute byte counts for each varint length (reused for early-exit checks)
            // Use number_of_varints as vl to exclude any incomplete varint at the end
            size_t count2 = __riscv_vcpop_m_b8(m_second_bytes, number_of_varints);

            // Total bytes = sum of bytes per varint
            size_t number_of_bytes = number_of_varints + count2;

            // remove continuation bits (bit 7) from payload bytes
            vuint8m1_t b1 = __riscv_vand_vx_u8m1(first_bytes, 0x7F, number_of_varints);
            vuint8m1_t b2 = __riscv_vand_vx_u8m1(second_bytes, 0x7F, number_of_varints);

            // Build result in 32-bit
            // b1: bits 0-6
            vuint32m4_t result32 = __riscv_vzext_vf4_u32m4(b1, number_of_varints);

            // b2: bits 7-13 (shift by 7)
            result32 = __riscv_vadd_vv_u32m4_mu(m_second_bytes, result32, result32,
                                                __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(b2, number_of_varints), 7, number_of_varints),
                                                number_of_varints);

            size_t count3 = __riscv_vcpop_m_b8(m_third_bytes, number_of_varints);
            // Only process 3+ byte varints if any exist
            if (count3 > 0)
            {
                number_of_bytes += count3;
                vuint8m1_t b3 = __riscv_vand_vx_u8m1(third_bytes, 0x7F, number_of_varints);

                // b3: bits 14-20 (shift by 14)
                result32 = __riscv_vadd_vv_u32m4_mu(m_third_bytes, result32, result32,
                                                    __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(b3, number_of_varints), 14, number_of_varints),
                                                    number_of_varints);

                size_t count4 = __riscv_vcpop_m_b8(m_fourth_bytes, number_of_varints);
                // Only process 4+ byte varints if any exist
                if (count4 > 0)
                {
                    number_of_bytes += count4;
                    vuint8m1_t b4 = __riscv_vand_vx_u8m1(fourth_bytes, 0x7F, number_of_varints);

                    // b4: bits 21-27 (shift by 21)
                    result32 = __riscv_vadd_vv_u32m4_mu(m_fourth_bytes, result32, result32,
                                                        __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(b4, number_of_varints), 21, number_of_varints),
                                                        number_of_varints);

                    size_t count5 = __riscv_vcpop_m_b8(m_fifth_bytes, number_of_varints);
                    // Only process 5 byte varints if any exist
                    if (count5 > 0)
                    {
                        number_of_bytes += count5;
                        vuint8m1_t b5 = __riscv_vand_vx_u8m1(fifth_bytes, 0x7F, number_of_varints);

                        // b5: bits 28-31 (shift by 28)
                        result32 = __riscv_vadd_vv_u32m4_mu(m_fifth_bytes, result32, result32,
                                                            __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(b5, number_of_varints), 28, number_of_varints),
                                                            number_of_varints);
                    }
                }
            }

            // Store decoded varints
            __riscv_vse32_v_u32m4(output, result32, number_of_varints);

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
 * Optimized variant that performs first two bytes' integration in u16m2 instead of u32m4,
 * deferring the extension to 32-bit only when needed for 3+ byte varints. Uses vwmaccu for combining bytes.
 *
 * input: uint8_t pointer to the start of the compressed varints
 * output: decompressed 32-bit integers
 * length: size of the varints in bytes
 * returns: size_t number of decompressed integers
 */
size_t varint_decode_vecshift_u16(const uint8_t *input, size_t length, uint32_t *output)
{
    size_t processed = 0;

    size_t vl;

    while (length > 0)
    {
        vl = __riscv_vsetvl_e8m1(length);

        vuint8m1_t data_vec_u8 = __riscv_vle8_v_u8m1(input, vl);

        vbool8_t termination_mask = __riscv_vmsge_vx_i8m1_b8(__riscv_vreinterpret_v_u8m1_i8m1(data_vec_u8), 0, vl);

        uint8_t number_of_varints = __riscv_vcpop_m_b8(termination_mask, vl);

        // fast path. no continuation bits set
        if (number_of_varints == vl)
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

            // inspired by https://github.com/camel-cdr/rvv-bench/blob/main/vector-utf/8toN_gather.c
            vuint8m1_t v1 = __riscv_vslide1down_vx_u8m1(data_vec_u8, 0, vl);
            vuint8m1_t v2 = __riscv_vslide1down_vx_u8m1(v1, 0, vl);
            vuint8m1_t v3 = __riscv_vslide1down_vx_u8m1(v2, 0, vl);
            vuint8m1_t v4 = __riscv_vslide1down_vx_u8m1(v3, 0, vl);

            // every byte after a termination byte is as first byte
            vuint8m1_t v_prev = __riscv_vslide1up(data_vec_u8, 0, vl);
            vbool8_t m_first_bytes = __riscv_vmsleu_vx_u8m1_b8(v_prev, 0x7F, vl);

            // compress the slided input data vectors with the first_byte mask. That way we get the second, third, fourth and fifth byte after each first byte.
            vuint8m1_t first_bytes = __riscv_vcompress_vm_u8m1(data_vec_u8, m_first_bytes, vl);
            vuint8m1_t second_bytes = __riscv_vcompress_vm_u8m1(v1, m_first_bytes, vl);
            vuint8m1_t third_bytes = __riscv_vcompress_vm_u8m1(v2, m_first_bytes, vl);
            vuint8m1_t fourth_bytes = __riscv_vcompress_vm_u8m1(v3, m_first_bytes, vl);
            vuint8m1_t fifth_bytes = __riscv_vcompress_vm_u8m1(v4, m_first_bytes, vl);

            vbool8_t m_second_bytes = __riscv_vmsgtu_vx_u8m1_b8(first_bytes, 0x7F, vl);
            vbool8_t m_third_bytes = __riscv_vmand_mm_b8(m_second_bytes, __riscv_vmsgtu_vx_u8m1_b8(second_bytes, 0x7F, vl), vl);
            vbool8_t m_fourth_bytes = __riscv_vmand_mm_b8(m_third_bytes, __riscv_vmsgtu_vx_u8m1_b8(third_bytes, 0x7F, vl), vl);
            vbool8_t m_fifth_bytes = __riscv_vmand_mm_b8(m_fourth_bytes, __riscv_vmsgtu_vx_u8m1_b8(fourth_bytes, 0x7F, vl), vl);

            // remove continuation bits (bit 7) from payload bytes
            vuint8m1_t b1 = __riscv_vand_vx_u8m1(first_bytes, 0x7F, number_of_varints);
            vuint8m1_t b2 = __riscv_vand_vx_u8m1(second_bytes, 0x7F, number_of_varints);
            vuint8m1_t b3 = __riscv_vand_vx_u8m1(third_bytes, 0x7F, number_of_varints);
            vuint8m1_t b4 = __riscv_vand_vx_u8m1(fourth_bytes, 0x7F, number_of_varints);

            // Build result in 16-bit first (fits 14 bits for 2-byte varints)
            // b1: bits 0-6 b2: bits 7-13 (shift by 7, i.e., multiply by 128)
            vuint16m2_t result12 = __riscv_vwmaccu_vx_u16m2_mu(m_second_bytes, __riscv_vzext_vf2_u16m2(b1, number_of_varints), 128, b2,
                                                               number_of_varints);

            // b3: bits 0-6 b4: bits 7-13 (shift by 7, i.e., multiply by 128)
            vuint16m2_t result34 = __riscv_vwmaccu_vx_u16m2_mu(m_fourth_bytes, __riscv_vzext_vf2_u16m2(b3, number_of_varints), 128, b4,
                                                               number_of_varints);

            // shift result12 left by 14 (multiply with 16384) and add it to result34
            vuint32m4_t result1234 = __riscv_vwmaccu_vx_u32m4_mu(m_third_bytes, __riscv_vzext_vf2_u32m4(result12, number_of_varints), 16384, result34, number_of_varints);

            // Only process 5 byte varints if any exist
            // They should be rare, as they only contribute bits 28-31 to the result, and cannot utilize the same vwmaccu optimization. So we branch here.
            size_t count5 = __riscv_vcpop_m_b8(m_fifth_bytes, number_of_varints);
            if (count5 > 0)
            {
                vuint8m1_t b5 = __riscv_vand_vx_u8m1(fifth_bytes, 0x7F, number_of_varints);

                // b5: bits 28-31 (shift by 28)
                // vwmaccu cannot be used as the scalar shift value is too large here.
                result1234 = __riscv_vadd_vv_u32m4_mu(m_fifth_bytes, result1234, result1234,
                                                      __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(b5, number_of_varints), 28, number_of_varints),
                                                      number_of_varints);
            }

            // Compute byte counts for each varint length
            // Use number_of_varints as vl to exclude any incomplete varint at the end
            size_t count2 = __riscv_vcpop_m_b8(m_second_bytes, number_of_varints);
            size_t count3 = __riscv_vcpop_m_b8(m_third_bytes, number_of_varints);
            size_t count4 = __riscv_vcpop_m_b8(m_fourth_bytes, number_of_varints);

            // Total bytes = sum of bytes per varint
            size_t number_of_bytes = number_of_varints + count2 + count3 + count4 + count5;

            // Store decoded varints
            __riscv_vse32_v_u32m4(output, result1234, number_of_varints);

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
size_t varint_decode_vecshift_m2(const uint8_t *input, size_t length, uint32_t *output)
{
    size_t processed = 0;

    size_t vl;

    while (length > 0)
    {
        vl = __riscv_vsetvl_e8m2(length);

        vuint8m2_t data_vec_u8 = __riscv_vle8_v_u8m2((int8_t *)input, vl);

        vbool4_t termination_mask = __riscv_vmsge_vx_i8m2_b4(__riscv_vreinterpret_v_u8m2_i8m2(data_vec_u8), 0, vl);

        uint8_t number_of_varints = __riscv_vcpop_m_b4(termination_mask, vl);

        // fast path. no continuation bits set
        if (number_of_varints == vl)
        {
            // expand every byte to 32-bit lane and save to memory
            __riscv_vse32_v_u32m8(output, __riscv_vzext_vf4_u32m8(data_vec_u8, vl), vl);
            input += vl;
            length -= vl;
            output += vl;
            processed += vl;
        }
        else
        {

            vuint8m2_t v1 = __riscv_vslide1down_vx_u8m2(data_vec_u8, 0, vl);
            vuint8m2_t v2 = __riscv_vslide1down_vx_u8m2(v1, 0, vl);
            vuint8m2_t v3 = __riscv_vslide1down_vx_u8m2(v2, 0, vl);
            vuint8m2_t v4 = __riscv_vslide1down_vx_u8m2(v3, 0, vl);

            vuint8m2_t v_prev = __riscv_vslide1up(data_vec_u8, 0, vl);
            vbool4_t m_first_bytes = __riscv_vmsleu_vx_u8m2_b4(v_prev, 0x7F, vl);

            vuint8m2_t first_bytes = __riscv_vcompress_vm_u8m2(data_vec_u8, m_first_bytes, vl);
            vuint8m2_t second_bytes = __riscv_vcompress_vm_u8m2(v1, m_first_bytes, vl);
            vuint8m2_t third_bytes = __riscv_vcompress_vm_u8m2(v2, m_first_bytes, vl);
            vuint8m2_t fourth_bytes = __riscv_vcompress_vm_u8m2(v3, m_first_bytes, vl);
            vuint8m2_t fifth_bytes = __riscv_vcompress_vm_u8m2(v4, m_first_bytes, vl);

            vbool4_t m_second_bytes = __riscv_vmsgtu_vx_u8m2_b4(first_bytes, 0x7F, vl);
            vbool4_t m_third_bytes = __riscv_vmand_mm_b4(m_second_bytes, __riscv_vmsgtu_vx_u8m2_b4(second_bytes, 0x7F, vl), vl);
            vbool4_t m_fourth_bytes = __riscv_vmand_mm_b4(m_third_bytes, __riscv_vmsgtu_vx_u8m2_b4(third_bytes, 0x7F, vl), vl);
            vbool4_t m_fifth_bytes = __riscv_vmand_mm_b4(m_fourth_bytes, __riscv_vmsgtu_vx_u8m2_b4(fourth_bytes, 0x7F, vl), vl);

            // Compute byte counts for each varint length (reused for early-exit checks)
            // Use number_of_varints as vl to exclude any incomplete varint at the end
            size_t count2 = __riscv_vcpop_m_b4(m_second_bytes, number_of_varints);

            // Total bytes = sum of bytes per varint
            size_t number_of_bytes = number_of_varints + count2;

            // remove continuation bits (bit 7) from payload bytes
            vuint8m2_t b1 = __riscv_vand_vx_u8m2(first_bytes, 0x7F, number_of_varints);
            vuint8m2_t b2 = __riscv_vand_vx_u8m2(second_bytes, 0x7F, number_of_varints);

            // Build result in 32-bit
            // b1: bits 0-6
            vuint32m8_t result32 = __riscv_vzext_vf4_u32m8(b1, number_of_varints);

            // b2: bits 7-13 (shift by 7)
            result32 = __riscv_vadd_vv_u32m8_mu(m_second_bytes, result32, result32,
                                                __riscv_vsll_vx_u32m8(__riscv_vzext_vf4_u32m8(b2, number_of_varints), 7, number_of_varints),
                                                number_of_varints);

            size_t count3 = __riscv_vcpop_m_b4(m_third_bytes, number_of_varints);
            // Only process 3+ byte varints if any exist
            if (count3 > 0)
            {
                number_of_bytes += count3;
                vuint8m2_t b3 = __riscv_vand_vx_u8m2(third_bytes, 0x7F, number_of_varints);

                // b3: bits 14-20 (shift by 14)
                result32 = __riscv_vadd_vv_u32m8_mu(m_third_bytes, result32, result32,
                                                    __riscv_vsll_vx_u32m8(__riscv_vzext_vf4_u32m8(b3, number_of_varints), 14, number_of_varints),
                                                    number_of_varints);

                size_t count4 = __riscv_vcpop_m_b4(m_fourth_bytes, number_of_varints);
                // Only process 4+ byte varints if any exist
                if (count4 > 0)
                {
                    number_of_bytes += count4;
                    vuint8m2_t b4 = __riscv_vand_vx_u8m2(fourth_bytes, 0x7F, number_of_varints);

                    // b4: bits 21-27 (shift by 21)
                    result32 = __riscv_vadd_vv_u32m8_mu(m_fourth_bytes, result32, result32,
                                                        __riscv_vsll_vx_u32m8(__riscv_vzext_vf4_u32m8(b4, number_of_varints), 21, number_of_varints),
                                                        number_of_varints);

                    size_t count5 = __riscv_vcpop_m_b4(m_fifth_bytes, number_of_varints);
                    // Only process 5 byte varints if any exist
                    if (count5 > 0)
                    {
                        number_of_bytes += count5;
                        vuint8m2_t b5 = __riscv_vand_vx_u8m2(fifth_bytes, 0x7F, number_of_varints);

                        // b5: bits 28-31 (shift by 28)
                        result32 = __riscv_vadd_vv_u32m8_mu(m_fifth_bytes, result32, result32,
                                                            __riscv_vsll_vx_u32m8(__riscv_vzext_vf4_u32m8(b5, number_of_varints), 28, number_of_varints),
                                                            number_of_varints);
                    }
                }
            }

            // Store decoded varints
            __riscv_vse32_v_u32m8(output, result32, number_of_varints);

            // loop bookkeeping
            input += number_of_bytes;
            length -= number_of_bytes;
            output += number_of_varints;
            processed += number_of_varints;
        }
    }
    return processed;
}