# Vectorized Varint Decoding for RISC-V

A library for exploring the performance of decoders for variable-length integers (varints) using RISC-V Vector (RVV) extensions. This library provides multiple vectorized implementations that achieve 1.5-2x speedup over scalar decoding.

## Overview

Variable-length integers (varints) are a compact encoding format used in protocols like Protocol Buffers, where smaller values require fewer bytes. Each byte uses 7 bits for data and 1 continuation bit (MSB), allowing values from 1-5 bytes for 32-bit integers.

This library implements several vectorized decoding strategies optimized for RISC-V processors with the Vector extension, along with a scalar baseline for comparison.

## Implementations

| Implementation | Description |
|---------------|-------------|
| `varint_decode_scalar` | Scalar baseline based on Protocol Buffers implementation |
| `varint_decode_maskshift` | RVV mask-based compression with byte shifting (m1/m2 variants) |
| `varint_decode_vecshift` | Vector slides and selective processing |
| `varint_decode_masked_vbyte` | Lookup table-based decoder with vector gather operations |

## Requirements

### Hardware
- RISC-V 64-bit processor with Vector extension (RVV 1.0)
- Tested on Spacemit X60 CPU

### Software
- RISC-V GCC toolchain (native or cross-compiler)
- CMake 3.13+
- Google Benchmark (included as git submodule)

## Building

### Clone with Submodules

```bash
git clone --recursive https://github.com/your-repo/varint_rvv.git
cd varint_rvv
```

If already cloned without submodules:
```bash
git submodule update --init --recursive
```

### Native Build (on RISC-V hardware)

```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLE=1 -DBUILD_BENCHMARK=1
make -j$(nproc)
```

### Cross-Compilation

```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=riscv64-unknown-linux-gnu-g++ \
    -DCMAKE_SYSTEM_NAME=Generic \
    -DBUILD_EXAMPLE=1 \
    -DBUILD_BENCHMARK=1 \
    -DBENCHMARK_ENABLE_WERROR=0
make -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_EXAMPLE` | OFF | Build the example executable |
| `BUILD_BENCHMARK` | OFF | Build the Google Benchmark suite |
| `BENCHMARK_ENABLE_WERROR` | ON | Treat warnings as errors in benchmark |

## Benchmarking

### Running Benchmarks

```bash
./build/benchmark
```

### Benchmark Configuration

The benchmark tests different implementations against varying varint distributions:

| Distribution | 1-byte | 2-byte | 3-byte | 4-byte | 5-byte |
|-------------|--------|--------|--------|--------|--------|
| Heavily Skewed | 95% | 3% | 1% | 1% | 0% |
| Mixed | 90% | 4% | 3% | 2% | 1% |
| Diverse | 81% | 7% | 6% | 4% | 2% |

Input sizes range from 1 KB to 4 MB, testing both cache-resident and memory-bound scenarios.

### Sample Results

Benchmark results on RISC-V hardware with RVV support:

| Size | VecShift (Mi/s) | Scalar (Mi/s) | Speedup |
|------|-----------------|---------------|---------|
| 1 KB | 459.4 | 243.7 | 1.89x |
| 4 KB | 465.8 | 232.3 | 2.01x |
| 64 KB | 327.0 | 220.2 | 1.49x |
| 1 MB | 309.1 | 206.5 | 1.50x |

*Results for heavily skewed distribution (95% single-byte varints)*

## Project Structure

```
varint_rvv/
├── CMakeLists.txt              # Build configuration
├── lib/
│   ├── include/
│   │   ├── libvarintrvv.h      # Public API header
│   │   └── utils.h             # Lookup tables and utilities
│   └── src/
│       ├── varint_encode.c     # Varint encoder
│       ├── varint_decode_scalar.c
│       ├── varint_decode_maskshift.c
│       ├── varint_decode_maskedvbyte.c
│       └── varint_decode_vecshift.c
├── example/
│   └── example.c               # Example usage
├── benchmark/
│   ├── benchmark.cc            # Google Benchmark harness
│   ├── plot_benchmark.py       # Results visualization
│   └── benchmark_results.txt   # Sample results
└── submodules/
    └── google-benchmark/       # Benchmark library
```

## Varint Encoding Format

| Value Range | Bytes | Encoding |
|------------|-------|----------|
| 0 - 127 | 1 | `0xxxxxxx` |
| 128 - 16,383 | 2 | `1xxxxxxx 0xxxxxxx` |
| 16,384 - 2,097,151 | 3 | `1xxxxxxx 1xxxxxxx 0xxxxxxx` |
| 2,097,152 - 268,435,455 | 4 | `1xxxxxxx 1xxxxxxx 1xxxxxxx 0xxxxxxx` |
| 268,435,456 - 4,294,967,295 | 5 | `1xxxxxxx 1xxxxxxx 1xxxxxxx 1xxxxxxx 0xxxxxxx` |

Each byte uses 7 bits for data (LSB first) and the MSB as a continuation flag (1 = more bytes follow, 0 = final byte).

## References

- [RISC-V Vector Extension Specification](https://github.com/riscv/riscv-v-spec)
- [Protocol Buffers Encoding](https://developers.google.com/protocol-buffers/docs/encoding)
- [Masked VByte Paper](https://arxiv.org/abs/1503.07387)
