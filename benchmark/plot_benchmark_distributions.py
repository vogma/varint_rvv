#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from benchmark_results.txt
message_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

# Distribution 1: 20, 20, 20, 20, 20 (uniform)
dist1_vecshift = [275.4, 275.327, 274.574, 274.007, 274.367, 274.172, 272.518, 268.899, 268.662, 268.039, 267.815]
dist1_masked_vbyte = [199.313, 197.858, 188.903, 187.15, 186.948, 186.669, 182.767, 179.714, 179.67, 179.163, 178.551]
dist1_scalar = [223.688, 213.63, 210.97, 205.69, 204.053, 203.993, 203.565, 203.657, 204.534, 204.78, 204.204]

# Distribution 2: 90, 4, 3, 2, 1 (heavily skewed to 1-byte)
dist2_vecshift = [344.979, 344.414, 346.54, 337.712, 335.461, 318.737, 305.884, 263.452, 259.042, 256.614, 254.255]
dist2_masked_vbyte = [248.655, 259.484, 233.718, 202.632, 203.492, 204.164, 204.011, 182.641, 164.146, 163.148, 161.957]
dist2_scalar = [290.375, 265.458, 250.919, 231.623, 232.739, 233.222, 233.736, 230.86, 234.757, 235.225, 235.837]

# Distribution 3: 81, 7, 6, 5, 1 (mostly 1-byte)
dist3_vecshift = [331.29, 326.029, 327.171, 324.718, 320.533, 304.617, 292.74, 261.324, 250.101, 245.382, 244.458]
dist3_masked_vbyte = [203.489, 196.172, 181.526, 167.893, 166.694, 166.307, 166.003, 156.387, 155.67, 155.17, 154.059]
dist3_scalar = [270.542, 230.279, 219.581, 208.166, 208.764, 207.86, 207.5, 208.521, 209.973, 209.961, 209.7]

# Distribution 4: 72, 13, 9, 5, 1 (moderate skew)
dist4_vecshift = [325.423, 322.813, 326.614, 321.605, 319.619, 308.878, 298.36, 267.054, 257.097, 250.789, 246.288]
dist4_masked_vbyte = [209.63, 203.41, 187.214, 172.367, 171.589, 170.419, 170.246, 158.745, 158.403, 157.943, 156.897]
dist4_scalar = [240.217, 213.151, 200.265, 189.756, 189.603, 188.428, 187.563, 189.605, 189.74, 189.784, 189.233]

distributions = [

    {
        'name': 'Skewed 1-byte (90-4-3-2-1)',
        'vecshift': dist2_vecshift,
        'masked_vbyte': dist2_masked_vbyte,
        'scalar': dist2_scalar,
    },
    {
        'name': 'Mostly 1-byte (81-7-6-5-1)',
        'vecshift': dist3_vecshift,
        'masked_vbyte': dist3_masked_vbyte,
        'scalar': dist3_scalar,
    },
    {
        'name': 'Moderate skew (72-13-9-5-1)',
        'vecshift': dist4_vecshift,
        'masked_vbyte': dist4_masked_vbyte,
        'scalar': dist4_scalar,
    },
        {
        'name': 'Uniform (20-20-20-20-20)',
        'vecshift': dist1_vecshift,
        'masked_vbyte': dist1_masked_vbyte,
        'scalar': dist1_scalar,
    }
]

def format_size(x, pos):
    if x >= 1048576:
        return f'{int(x/1048576)}M'
    elif x >= 1024:
        return f'{int(x/1024)}K'
    return str(int(x))

# Create 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, dist in enumerate(distributions):
    ax = axes[idx]

    ax.plot(message_sizes, dist['vecshift'], 'o-',
            label='varint_rvv', color='#2563eb', linewidth=1.5, markersize=4)
    ax.plot(message_sizes, dist['scalar'], 's-',
            label='scalar', color='#dc2626', linewidth=1.5, markersize=4)
    # ax.plot(message_sizes, dist['masked_vbyte'], '^-',
    #         label='masked_vbyte', color='#059669', linewidth=1.5, markersize=4)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('Throughput (MiB/s)')
    ax.set_title(dist['name'])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_size))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    # Set consistent y-axis range across all plots
    ax.set_ylim(100, 400)

plt.suptitle('Varint Decoding Throughput by Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('varint_throughput_distributions.svg', format='svg', bbox_inches='tight')
plt.savefig('varint_throughput_distributions.pdf', format='pdf', bbox_inches='tight')
plt.savefig('varint_throughput_distributions.png', format='png', dpi=150, bbox_inches='tight')
print("Saved: varint_throughput_distributions.svg / .pdf / .png")
plt.show()
