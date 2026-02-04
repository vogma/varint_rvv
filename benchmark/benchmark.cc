#include <benchmark/benchmark.h>
#include <libvarintrvv.h>
#include <cstdint>
#include <vector>
#include <random>
#include <limits.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C"
{
    size_t varint_decode_scalar(const uint8_t *input, int length, uint32_t *output);
    size_t varint_decode_vecshift(const uint8_t *input, size_t length, uint32_t *output);
    size_t varint_decode_vecshift_m2(const uint8_t *input, size_t length, uint32_t *output);
    size_t varint_decode_masked_vbyte(const uint8_t *input, size_t length, uint32_t *output);
    size_t varint_decode_masked_vbyte_opt(const uint8_t *input, size_t length, uint32_t *output);
    size_t vbyte_encode(const uint32_t *in, size_t length, uint8_t *bout);
}

struct Dataset
{
    std::vector<uint8_t> input;
    std::vector<uint32_t> output;
};

// Varint byte ranges:
// 1 byte: 0 - 127
// 2 bytes: 128 - 16383
// 3 bytes: 16384 - 2097151
// 4 bytes: 2097152 - 268435455
// 5 bytes: 268435456 - 4294967295

static std::vector<uint8_t> generate_test_data(size_t num_values, uint32_t seed,
                                               int pct_1byte, int pct_2byte,
                                               int pct_3byte, int pct_4byte,
                                               int pct_5byte)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pct_dist(0, 99);

    // Ranges for each byte length
    std::uniform_int_distribution<uint32_t> dist_1byte(0, 127);
    std::uniform_int_distribution<uint32_t> dist_2byte(128, 16383);
    std::uniform_int_distribution<uint32_t> dist_3byte(16384, 2097151);
    std::uniform_int_distribution<uint32_t> dist_4byte(2097152, 268435455);
    std::uniform_int_distribution<uint32_t> dist_5byte(268435456, UINT32_MAX);

    std::vector<uint32_t> values(num_values);

    // Cumulative thresholds
    int thresh_1 = pct_1byte;
    int thresh_2 = thresh_1 + pct_2byte;
    int thresh_3 = thresh_2 + pct_3byte;
    int thresh_4 = thresh_3 + pct_4byte;

    for (size_t i = 0; i < num_values; ++i)
    {
        int roll = pct_dist(rng);
        if (roll < thresh_1)
            values[i] = dist_1byte(rng);
        else if (roll < thresh_2)
            values[i] = dist_2byte(rng);
        else if (roll < thresh_3)
            values[i] = dist_3byte(rng);
        else if (roll < thresh_4)
            values[i] = dist_4byte(rng);
        else
            values[i] = dist_5byte(rng);
    }

    // Encode to varints (max 5 bytes per value)
    std::vector<uint8_t> encoded(num_values * 5);
    size_t encoded_size = vbyte_encode(values.data(), num_values, encoded.data());
    encoded.resize(encoded_size);

    return encoded;
}

static Dataset make_dataset(size_t num_values, uint32_t seed,
                            int pct_1byte = 100, int pct_2byte = 0,
                            int pct_3byte = 0, int pct_4byte = 0,
                            int pct_5byte = 0)
{
    Dataset ds;
    ds.input = generate_test_data(num_values, seed, pct_1byte, pct_2byte,
                                  pct_3byte, pct_4byte, pct_5byte);
    ds.output.resize(num_values);
    return ds;
}

class PerfCounter
{
public:
    PerfCounter(uint64_t event_type, uint64_t event_config)
    {
        struct perf_event_attr pe = {};
        pe.type = event_type;
        pe.size = sizeof(pe);
        pe.config = event_config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;

        fd_ = syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
    }

    ~PerfCounter()
    {
        if (fd_ >= 0)
            close(fd_);
    }

    bool valid() const { return fd_ >= 0; }

    void start()
    {
        if (fd_ >= 0)
        {
            ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
        }
    }

    uint64_t stop()
    {
        uint64_t count = 0;
        if (fd_ >= 0)
        {
            ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
            read(fd_, &count, sizeof(count));
        }
        return count;
    }

private:
    int fd_ = -1;
};

template <auto DecoderFn, int P1, int P2, int P3, int P4, int P5>
static void BM_decode(benchmark::State &state)
{
    const size_t num_values = static_cast<size_t>(state.range(0));
    auto ds = make_dataset(num_values, 12345, P1, P2, P3, P4, P5);

    size_t total_ints = 0;
    uint64_t total_instructions = 0;

    PerfCounter insn_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);

    insn_counter.start();
    for (auto _ : state)
    {
        size_t n = DecoderFn(ds.input.data(), ds.input.size(), ds.output.data());
        total_ints += n;

        benchmark::DoNotOptimize(n);
        benchmark::DoNotOptimize(ds.output.data());
        benchmark::ClobberMemory();
    }
    total_instructions = insn_counter.stop();

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(ds.input.size()));
    state.SetItemsProcessed(int64_t(total_ints));

    if (insn_counter.valid())
    {
        state.counters["insn/byte"] = benchmark::Counter(
            double(total_instructions) / double(ds.input.size() * state.iterations()),
            benchmark::Counter::kAvgThreads);
        state.counters["insn/int"] = benchmark::Counter(
            double(total_instructions) / double(total_ints),
            benchmark::Counter::kAvgThreads);
    }
}

BENCHMARK_TEMPLATE(BM_decode, varint_decode_masked_vbyte_opt, 100, 0, 0, 0, 0)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_vecshift, 100, 0, 0, 0, 0)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_scalar, 100, 0, 0, 0, 0)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

BENCHMARK_TEMPLATE(BM_decode, varint_decode_masked_vbyte_opt,  95,3,1,1,0)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_vecshift,  95,3,1,1,0)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);
// BENCHMARK_TEMPLATE(BM_decode, varint_decode_vecshift_m2,  95,3,1,1,1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
// BENCHMARK_TEMPLATE(BM_decode, varint_decode_m2,  95,3,1,1,1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_scalar,  95,3,1,1,0)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);

// Distribution: 20% each byte length (uniform)
BENCHMARK_TEMPLATE(BM_decode, varint_decode_masked_vbyte_opt, 20, 20, 20, 20, 20)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_scalar, 20, 20, 20, 20, 20)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_vecshift, 20, 20, 20, 20, 20)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

// Distribution: 90% 1-byte, 4% 2-byte, 3% 3-byte, 2% 4-byte, 1% 5-byte (small values)
BENCHMARK_TEMPLATE(BM_decode, varint_decode_masked_vbyte_opt, 90, 4, 3, 2, 1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_vecshift, 90, 4, 3, 2, 1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_scalar, 90, 4, 3, 2, 1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

// Distribution: 81% 1-byte, 7% 2-byte, 6% 3-byte, 5% 4-byte, 1% 5-byte (mixed)
BENCHMARK_TEMPLATE(BM_decode, varint_decode_masked_vbyte_opt, 81, 7, 6, 5, 1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_vecshift, 81, 7, 6, 5, 1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_scalar, 81, 7, 6, 5, 1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

// Distribution: 72% 1-byte, 13% 2-byte, 9% 3-byte, 5% 4-byte, 1% 5-byte (mixed)
BENCHMARK_TEMPLATE(BM_decode, varint_decode_masked_vbyte_opt, 72, 13, 9, 5, 1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_vecshift, 72, 13, 9, 5, 1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode_scalar, 72, 13, 9, 5, 1)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

BENCHMARK_MAIN();