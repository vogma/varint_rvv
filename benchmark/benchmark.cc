#include <benchmark/benchmark.h>
#include <libvarintrvv.h>
#include <cstdint>
#include <vector>
#include <random>

extern "C"
{
    uint64_t varint_decode(uint8_t *input, uint32_t *output, size_t length);
}

struct Dataset
{
    std::vector<uint8_t> input;
    std::vector<uint32_t> output;
};

static Dataset make_dataset(size_t input_bytes, uint32_t seed)
{
    Dataset ds;
    ds.input.resize(input_bytes);
    ds.output.resize(input_bytes);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < input_bytes; ++i)
        ds.input[i] = static_cast<uint8_t>(dist(rng));

    return ds;
}

template <auto DecoderFn>
static void BM_decode(benchmark::State &state)
{
    const size_t input_bytes = static_cast<size_t>(state.range(0));
    auto ds = make_dataset(input_bytes, 12345);

    for (auto _ : state)
    {

        size_t n = DecoderFn(ds.input.data(), ds.output.data(), ds.input.size());

        benchmark::DoNotOptimize(n);
        benchmark::DoNotOptimize(ds.output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(ds.input.size()));
}

// BENCHMARK_TEMPLATE(BM_decode, decode_varints_scalar)->RangeMultiplier(2)->Range(1<<10, 1<<20);
BENCHMARK_TEMPLATE(BM_decode, varint_decode)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

BENCHMARK_MAIN();