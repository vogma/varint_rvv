#include "libvarintrvv.h"
#include <stdio.h>

size_t varint_decode_vecshift(const uint8_t *data, size_t length, uint32_t *output)
{
    size_t processed = 0;

    size_t vl;

    while (length > 0)
    {
        vl = __riscv_vsetvl_e8m1(length);

        vuint8m1_t input = __riscv_vle8_v_u8m1(data, vl);

        // mask set when element has termination bit (MSB==0) set
        vbool8_t termination_mask = __riscv_vmsleu(input, 0x7F, vl);

        // popcount on termination mask tells us number of complete varints in register, as bytes with termination bit set are at the last position in a varint.
        size_t num_varints = __riscv_vcpop(termination_mask, vl);

        // fast path. No continuation bits (MSB==1) set
        if (num_varints == vl)
        {
            // expand every byte to 32-bit lane and save to memory
            __riscv_vse32_v_u32m4(output, __riscv_vzext_vf4(input, vl), vl);

            data += vl;
            length -= vl;
            output += vl;
            processed += vl;
        }
        else
        {

            // inspired by https://github.com/camel-cdr/rvv-bench/blob/main/vector-utf/8toN_gather.c
            vuint8m1_t v1 = __riscv_vslide1down(input, 0, vl);
            vuint8m1_t v2 = __riscv_vslide1down(v1, 0, vl);

            // every byte after a termination byte is as first byte
            vuint8m1_t v_prev = __riscv_vslide1up(input, 0, vl);
            vbool8_t m_first_bytes = __riscv_vmsleu(v_prev, 0x7F, vl);

            // compress the slided input data vectors with the first_byte mask. That way we get the second, third, fourth and fifth byte after each first byte.
            vuint8m1_t first_bytes = __riscv_vcompress(input, m_first_bytes, vl);
            vuint8m1_t second_bytes = __riscv_vcompress(v1, m_first_bytes, vl);

            vbool8_t m_second_bytes = __riscv_vmsgtu(first_bytes, 0x7F, vl);
            vbool8_t m_third_bytes = __riscv_vmand(m_second_bytes, __riscv_vmsgtu(second_bytes, 0x7F, vl), vl);

            // remove continuation bits (bit 7) from payload bytes
            vuint8m1_t b1 = __riscv_vand(first_bytes, 0x7F, vl);
            vuint8m1_t b2 = __riscv_vand(second_bytes, 0x7F, vl);

            // Build result in 16-bit first (fits 14 bits for 2-byte varints)
            // b1: bits 0-6 b2: bits 7-13 (shift by 7, i.e., multiply by 128)
            vuint16m2_t result12 = __riscv_vwmaccu_mu(m_second_bytes, __riscv_vzext_vf2(b1, vl), 128, b2,
                                                      vl);

            // Compute byte counts for each varint length
            // Use num_varints as vl to exclude any incomplete varint at the end
            size_t count2 = __riscv_vcpop(m_second_bytes, num_varints);
            size_t count3 = __riscv_vcpop(m_third_bytes, num_varints);

            // Total bytes = sum of bytes per varint
            size_t number_of_bytes = num_varints + count2 + count3;

            if (count3 == 0)
            {
                __riscv_vse32_v_u32m4(output, __riscv_vzext_vf2(result12, vl), num_varints);
            }
            else
            {
                vuint8m1_t v3 = __riscv_vslide1down(v2, 0, vl);
                vuint8m1_t v4 = __riscv_vslide1down(v3, 0, vl);

                vuint8m1_t third_bytes = __riscv_vcompress(v2, m_first_bytes, vl);
                vuint8m1_t fourth_bytes = __riscv_vcompress(v3, m_first_bytes, vl);

                vbool8_t m_fourth_bytes = __riscv_vmand(m_third_bytes, __riscv_vmsgtu(third_bytes, 0x7F, vl), vl);
                vbool8_t m_fifth_bytes = __riscv_vmand(m_fourth_bytes, __riscv_vmsgtu(fourth_bytes, 0x7F, vl), vl);

                size_t count4 = __riscv_vcpop(m_fourth_bytes, num_varints);
                size_t count5 = __riscv_vcpop(m_fifth_bytes, num_varints);

                number_of_bytes += count4;

                vuint8m1_t b3 = __riscv_vand(third_bytes, 0x7F, vl);
                vuint8m1_t b4 = __riscv_vand(fourth_bytes, 0x7F, vl);

                // b3: bits 0-6 b4: bits 7-13 (shift by 7, i.e., multiply by 128)
                vuint16m2_t result34 = __riscv_vwmaccu_mu(m_fourth_bytes, __riscv_vzext_vf2(b3, vl), 128, b4, vl);

                // shift result12 left by 14 (multiply with 16384) and add it to result34
                vuint32m4_t result1234 = __riscv_vwmaccu_mu(m_third_bytes, __riscv_vzext_vf2(result12, vl), 16384, result34, vl);

                if (count5 > 0)
                {
                    number_of_bytes += count5;

                    vuint8m1_t fifth_bytes = __riscv_vcompress(v4, m_first_bytes, vl);

                    vuint8m1_t b5 = __riscv_vand(fifth_bytes, 0x7F, vl);

                    // b5: bits 28-31 (shift by 28)
                    // vwmaccu cannot be used as the scalar shift value is too large.
                    result1234 = __riscv_vadd_mu(m_fifth_bytes, result1234, result1234, __riscv_vsll(__riscv_vzext_vf4(b5, vl), 28, vl), vl);
                }

                // Store decoded varints
                __riscv_vse32_v_u32m4(output, result1234, num_varints);
            }

            data += number_of_bytes;
            length -= number_of_bytes;
            output += num_varints;
            processed += num_varints;
        }
    }
    return processed;
}

size_t varint_decode_vecshift_test_m2(const uint8_t *data, size_t length, uint32_t *output)
{
    size_t processed = 0;

    size_t vl;

    while (length > 0)
    {
        vl = __riscv_vsetvl_e8m2(length);

        vuint8m2_t input = __riscv_vle8_v_u8m2(data, vl);

        // mask set when element has termination bit (MSB==0) set
        vbool4_t termination_mask = __riscv_vmsleu(input, 0x7F, vl);

        // popcount on termination mask tells us number of complete varints in register, as bytes with termination bit set are at the last position in a varint.
        size_t num_varints = __riscv_vcpop(termination_mask, vl);

        // fast path. No continuation bits (MSB==1) set
        if (num_varints == vl)
        {
            // expand every byte to 32-bit lane and save to memory
            __riscv_vse32_v_u32m8(output, __riscv_vzext_vf4(input, vl), vl);

            data += vl;
            length -= vl;
            output += vl;
            processed += vl;
        }
        else
        {

            // inspired by https://github.com/camel-cdr/rvv-bench/blob/main/vector-utf/8toN_gather.c
            vuint8m2_t v1 = __riscv_vslide1down(input, 0, vl);
            vuint8m2_t v2 = __riscv_vslide1down(v1, 0, vl);

            // every byte after a termination byte is as first byte
            vuint8m2_t v_prev = __riscv_vslide1up(input, 0, vl);
            vbool4_t m_first_bytes = __riscv_vmsleu(v_prev, 0x7F, vl);

            // compress the slided input data vectors with the first_byte mask. That way we get the second, third, fourth and fifth byte after each first byte.
            vuint8m2_t first_bytes = __riscv_vcompress(input, m_first_bytes, vl);
            vuint8m2_t second_bytes = __riscv_vcompress(v1, m_first_bytes, vl);

            vbool4_t m_second_bytes = __riscv_vmsgtu(first_bytes, 0x7F, vl);
            vbool4_t m_third_bytes = __riscv_vmand(m_second_bytes, __riscv_vmsgtu(second_bytes, 0x7F, vl), vl);

            // remove continuation bits (bit 7) from payload bytes
            vuint8m2_t b1 = __riscv_vand(first_bytes, 0x7F, num_varints);
            vuint8m2_t b2 = __riscv_vand(second_bytes, 0x7F, num_varints);

            // Build result in 16-bit first (fits 14 bits for 2-byte varints)
            // b1: bits 0-6 b2: bits 7-13 (shift by 7, i.e., multiply by 128)
            vuint16m4_t result12 = __riscv_vwmaccu(m_second_bytes, __riscv_vzext_vf2(b1, num_varints), 128, b2,
                                                   num_varints);

            // Compute byte counts for each varint length
            // Use num_varints as vl to exclude any incomplete varint at the end
            size_t count2 = __riscv_vcpop(m_second_bytes, num_varints);
            size_t count3 = __riscv_vcpop(m_third_bytes, num_varints);

            // Total bytes = sum of bytes per varint
            size_t number_of_bytes = num_varints + count2 + count3;

            if (count3 == 0)
            {
                __riscv_vse32_v_u32m8(output, __riscv_vzext_vf2(result12, number_of_bytes), num_varints);
            }
            else
            {
                vuint8m2_t v3 = __riscv_vslide1down(v2, 0, vl);
                vuint8m2_t v4 = __riscv_vslide1down(v3, 0, vl);

                vuint8m2_t third_bytes = __riscv_vcompress(v2, m_first_bytes, vl);
                vuint8m2_t fourth_bytes = __riscv_vcompress(v3, m_first_bytes, vl);

                vbool4_t m_fourth_bytes = __riscv_vmand(m_third_bytes, __riscv_vmsgtu(third_bytes, 0x7F, vl), vl);
                vbool4_t m_fifth_bytes = __riscv_vmand(m_fourth_bytes, __riscv_vmsgtu(fourth_bytes, 0x7F, vl), vl);

                size_t count4 = __riscv_vcpop(m_fourth_bytes, num_varints);
                size_t count5 = __riscv_vcpop(m_fifth_bytes, num_varints);

                number_of_bytes += count4;

                vuint8m2_t b3 = __riscv_vand(third_bytes, 0x7F, num_varints);
                vuint8m2_t b4 = __riscv_vand(fourth_bytes, 0x7F, num_varints);

                // b3: bits 0-6 b4: bits 7-13 (shift by 7, i.e., multiply by 128)
                vuint16m4_t result34 = __riscv_vwmaccu(m_fourth_bytes, __riscv_vzext_vf2_u16m4(b3, num_varints), 128, b4,
                                                       num_varints);

                // shift result12 left by 14 (multiply with 16384) and add it to result34
                vuint32m8_t result1234 = __riscv_vwmaccu(m_third_bytes, __riscv_vzext_vf2(result12, num_varints), 16384, result34, num_varints);

                if (count5 > 0)
                {
                    number_of_bytes += count5;

                    vuint8m2_t fifth_bytes = __riscv_vcompress(v4, m_first_bytes, vl);

                    vuint8m2_t b5 = __riscv_vand(fifth_bytes, 0x7F, num_varints);

                    // b5: bits 28-31 (shift by 28)
                    // vwmaccu cannot be used as the scalar shift value is too large.
                    result1234 = __riscv_vadd_vv_u32m8_mu(m_fifth_bytes, result1234, result1234, __riscv_vsll(__riscv_vzext_vf4(b5, num_varints), 28, num_varints), num_varints);
                }

                // Store decoded varints
                __riscv_vse32_v_u32m8(output, result1234, num_varints);
            }

            data += number_of_bytes;
            length -= number_of_bytes;
            output += num_varints;
            processed += num_varints;
        }
    }
    return processed;
}