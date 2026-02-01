#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from benchmark_results.txt
message_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

# bytes_per_second in Mi/s
vecshift_throughput = [459.449, 468.206, 465.835, 446.667, 443.561, 397.822, 327.019, 305.541, 293.888, 306.876, 309.066]
scalar_throughput = [243.737, 238.876, 232.253, 221.771, 225.196, 222.665, 220.161, 214.438, 212.605, 212.767, 212.816]

# Calculate speedup ratio
speedup = [v / s for v, s in zip(vecshift_throughput, scalar_throughput)]

# Format x-axis ticks to show KB/MB
def format_size(x, pos):
    if x >= 1048576:
        return f'{int(x/1048576)}M'
    elif x >= 1024:
        return f'{int(x/1024)}K'
    return str(int(x))

# --- Figure 1: Throughput comparison ---
fig1, ax1 = plt.subplots(figsize=(5, 3.5))

ax1.plot(message_sizes, vecshift_throughput, 'o-', label='Vectorized (RVV)', color='#2563eb', linewidth=1.5, markersize=4)
ax1.plot(message_sizes, scalar_throughput, 's--', label='Scalar', color='#dc2626', linewidth=1.5, markersize=4)

ax1.set_xscale('log', base=2)
ax1.set_xlabel('Message Size (bytes)')
ax1.set_ylabel('Throughput (MiB/s)')
ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_size))
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_ylim(150, 500)

plt.tight_layout()
plt.savefig('varint_throughput.svg', format='svg', bbox_inches='tight')
plt.savefig('varint_throughput.pdf', format='pdf', bbox_inches='tight')
print("Saved: varint_throughput.svg / .pdf")

# --- Figure 2: Speedup ratio ---
fig2, ax2 = plt.subplots(figsize=(5, 3))

ax2.plot(message_sizes, speedup, 'o-', color='#059669', linewidth=1.5, markersize=5)

# Add horizontal reference line at 1.0x
ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)

# Add vertical line at L1d cache boundary (32KB)
ax2.axvline(x=32768, color='#f59e0b', linestyle='--', linewidth=1.2, alpha=0.8)
ax2.text(32768 * 1.15, max(speedup) - 0.05, 'L1d (32KB)', fontsize=8, color='#f59e0b', va='top')

ax2.set_xscale('log', base=2)
ax2.set_xlabel('Message Size (bytes)')
ax2.set_ylabel('Speedup (RVV / Scalar)')
ax2.xaxis.set_major_formatter(plt.FuncFormatter(format_size))
ax2.grid(True, linestyle='--', alpha=0.3)

# Y-axis range
ax2.set_ylim(1.0, 2.1)

# Annotate compute-bound vs memory-bound regions
ax2.annotate('Compute\nbound', xy=(4096, 1.95), fontsize=8, ha='center', color='#2563eb')
ax2.annotate('Memory\nbound', xy=(262144, 1.42), fontsize=8, ha='center', color='#dc2626')

plt.tight_layout()
plt.savefig('varint_speedup.svg', format='svg', bbox_inches='tight')
plt.savefig('varint_speedup.pdf', format='pdf', bbox_inches='tight')
print("Saved: varint_speedup.svg / .pdf")
