#include "libvarintrvv.h"
#include <stdio.h>

// void print_vuint8m1(vuint8m1_t vec, size_t vl)
// {
//     uint8_t buf[256];
//     __riscv_vse8_v_u8m1(buf, vec, vl);
//     printf("vuint8m1[%zu]: ", vl);
//     for (size_t i = 0; i < vl; i++)
//     {
//         printf("%02x ", buf[i]);
//     }
//     printf("\n");
// }

// void print_vbool8(vbool8_t mask, size_t vl)
// {
//     uint8_t buf[32];
//     __riscv_vsm_v_b8(buf, mask, vl);
//     printf("vbool8[%zu]: ", vl);
//     for (size_t i = 0; i < vl; i++)
//     {
//         printf("%d", (buf[i / 8] >> (i % 8)) & 1);
//     }
//     printf("\n");
// }

// void print_masked_vuint8m1(const char *label, vuint8m1_t vec, vbool8_t mask, size_t vl)
// {
//     uint8_t vec_buf[256];
//     uint8_t mask_buf[32];
//     __riscv_vse8_v_u8m1(vec_buf, vec, vl);
//     __riscv_vsm_v_b8(mask_buf, mask, vl);

//     size_t count = __riscv_vcpop_m_b8(mask, vl);
//     printf("%s (valid=%zu): ", label, count);
//     for (size_t i = 0; i < vl; i++)
//     {
//         int bit = (mask_buf[i / 8] >> (i % 8)) & 1;
//         if (bit)
//         {
//             printf("%02x ", vec_buf[i]);
//         }
//         else
//         {
//             printf("-- ");
//         }
//     }
//     printf("\n");
// }

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
 * returns: uint64_t number of decompressed integers
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

            vuint8m1_t v1 = __riscv_vslide1down_vx_u8m1(data_vec_u8, 0, vl);
            vuint8m1_t v2 = __riscv_vslide1down_vx_u8m1(v1, 0, vl);
            vuint8m1_t v3 = __riscv_vslide1down_vx_u8m1(v2, 0, vl);
            vuint8m1_t v4 = __riscv_vslide1down_vx_u8m1(v3, 0, vl);

            vuint8m1_t v_prev = __riscv_vslide1up(data_vec_u8, 0, vl);
            vbool8_t m_first_bytes = __riscv_vmsleu_vx_u8m1_b8(v_prev, 0x7F, vl);

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