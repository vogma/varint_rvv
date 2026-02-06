#include "libvarintrvv.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* Simple PRNG */
typedef struct
{
    uint64_t x, y, z;
} URand;
static URand randState = {123, 456, 789};

static uint64_t urand(URand *r)
{
    uint64_t x = r->x, y = r->y, z = r->z;
    r->x = y;
    r->y = z;
    z = x ^ (x << 13);
    z ^= z >> 7;
    z ^= y ^ (y << 17);
    r->z = z;
    return z;
}

static uint64_t bench_urand(void) { return urand(&randState); }

/* Random helper that respects an upper bound */
static inline uint32_t rand_in_range(uint32_t low, uint32_t high_exclusive)
{
    return low + (uint32_t)(bench_urand() % (high_exclusive - low));
}

/* Generate a value that will encode to exactly len bytes */
static inline uint32_t random_value_for_length(uint8_t len)
{
    switch (len)
    {
    case 1:
        return (uint32_t)(bench_urand() & 0x7F);
    case 2:
        return rand_in_range(1u << 7, 1u << 14);
    case 3:
        return rand_in_range(1u << 14, 1u << 21);
    case 4:
        return rand_in_range(1u << 21, 1u << 28);
    default: /* len == 5 */
        return ((uint32_t)bench_urand()) | (1u << 28);
    }
}

/* Pick a varint length according to the distribution weights */
static inline uint8_t pick_length(uint8_t max_len, const int w[5])
{
    int total = 0;
    for (uint8_t i = 0; i < max_len; ++i)
        total += w[i];

    uint32_t r = (uint32_t)(bench_urand() % total);
    int acc = 0;
    for (uint8_t i = 0; i < max_len; ++i)
    {
        acc += w[i];
        if (r < (uint32_t)acc)
            return (uint8_t)(i + 1);
    }
    return max_len;
}

/* Determine varint length by scanning bytes */
static size_t get_varint_length(const uint8_t *p, const uint8_t *end)
{
    size_t len = 0;
    while (p < end && len < 5)
    {
        len++;
        if (!(*p & 0x80)) /* no continuation bit = last byte */
            return len;
        p++;
    }
    return len;
}

int main(void)
{
    /* Configuration */
    const size_t N = 1000;                   /* number of integers to test */
    const int weights[5] = {85, 5, 4, 3, 3}; /* even distribution: 20% each */

    printf("Varint Decode Test with Even Distribution\n");
    printf("==========================================\n\n");

    /* Allocate buffers */
    uint32_t *original_values = malloc(N * sizeof(uint32_t));
    uint8_t *encoded_data = malloc(N * 5); /* max 5 bytes per varint */
    uint32_t *decoded_values = malloc(N * sizeof(uint32_t));

    if (!original_values || !encoded_data || !decoded_values)
    {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    /* Generate original values with even distribution */
    size_t dist_counts[5] = {0};
    for (size_t i = 0; i < N; i++)
    {
        uint8_t len = pick_length(5, weights);
        original_values[i] = random_value_for_length(len);
        dist_counts[len - 1]++;
    }

    /* Print generation distribution */
    printf("Generated %zu values with distribution:\n", N);
    for (int i = 0; i < 5; i++)
    {
        printf("  %d-byte: %zu (%.1f%%)\n", i + 1, dist_counts[i],
               100.0 * dist_counts[i] / N);
    }
    printf("\n");

    /* Encode to varints */
    size_t encoded_length = vbyte_encode(original_values, N, encoded_data);
    printf("Encoded to %zu bytes (avg %.2f bytes/value)\n\n",
           encoded_length, (double)encoded_length / N);

    /* Verify encoded distribution by scanning the encoded data */
    size_t encoded_dist[5] = {0};
    size_t encoded_count = 0;
    const uint8_t *p = encoded_data;
    const uint8_t *end = encoded_data + encoded_length;
    while (p < end)
    {
        size_t len = get_varint_length(p, end);
        if (len == 0 || len > 5)
            break;
        encoded_dist[len - 1]++;
        encoded_count++;
        p += len;
    }

    printf("Encoded distribution verification:\n");
    for (int i = 0; i < 5; i++)
    {
        printf("  %d-byte: %zu (%.1f%%)\n", i + 1, encoded_dist[i],
               100.0 * encoded_dist[i] / encoded_count);
    }
    printf("\n");

    size_t decoded_count = varint_rvv_m2(encoded_data, encoded_length, decoded_values);
    printf("Decoded %zu integers from %zu bytes\n\n", decoded_count, encoded_length);

    /* Validate */
    int errors = 0;
    for (size_t i = 0; i < N; i++)
    {
        if (decoded_values[i] != original_values[i])
        {
            printf("ERROR at index %zu: expected %u, got %u\n",
                   i, original_values[i], decoded_values[i]);
            errors++;
            if (errors >= 10)
            {
                printf("... (stopping after 10 errors)\n");
                break;
            }
        }
    }

    /* Report summary */
    if (errors == 0)
    {
        printf("SUCCESS: All %zu values decoded correctly\n", N);
    }
    else
    {
        printf("FAILED: %d errors found\n", errors);
    }

    /* Cleanup */
    free(original_values);
    free(encoded_data);
    free(decoded_values);

    return errors > 0 ? 1 : 0;
}
