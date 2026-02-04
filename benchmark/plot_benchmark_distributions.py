#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from benchmark_results.txt
message_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

# Distribution 1: 20, 20, 20, 20, 20 (uniform)
dist1_vecshift = [364.999, 364.989, 363.549, 362.576, 362.369, 361.451, 355.785, 350.92, 348.863, 347.385, 346.574]
dist1_scalar = [208.728, 199.14, 196.662, 192.062, 190.929, 191.025, 189.39, 189.395, 190.367, 190.669, 190.345]

# Distribution 2: 90, 4, 3, 2, 1 (heavily skewed to 1-byte)
dist2_vecshift = [428.777, 422.526, 424.181, 419.926, 418.089, 378.157, 308.564, 298.234, 312.151, 323.213, 318.712]
dist2_scalar = [248.252, 230.971, 218.193, 204.575, 205.738, 206.088, 208.35, 204.506, 206.488, 207.644, 207.399]

# Distribution 3: 81, 7, 6, 5, 1 (mostly 1-byte)
dist3_vecshift = [419.867, 416.521, 419.117, 416.611, 412.901, 367.514, 324.904, 311.347, 327.003, 330.441, 331.104]
dist3_scalar = [230.589, 204.633, 195.64, 187.038, 187.482, 187.053, 187.123, 187.861, 188.324, 188.554, 188.195]

# Distribution 4: 72, 13, 9, 5, 1 (moderate skew)
dist4_vecshift = [417.946, 416.563, 420.422, 414.728, 412.518, 374.8, 347.055, 322.597, 334.768, 336.818, 334.601]
dist4_scalar = [211.638, 190.3, 179.69, 171.484, 171.642, 170.886, 170.653, 171.581, 171.989, 171.955, 171.684]

distributions = [

    {
        'name': 'Skewed 1-byte (90-4-3-2-1)',
        'vecshift': dist2_vecshift,
        'scalar': dist2_scalar,
    },
    {
        'name': 'Mostly 1-byte (81-7-6-5-1)',
        'vecshift': dist3_vecshift,
        'scalar': dist3_scalar,
    },
    {
        'name': 'Moderate skew (72-13-9-5-1)',
        'vecshift': dist4_vecshift,
        'scalar': dist4_scalar,
    },
        {
        'name': 'Uniform (20-20-20-20-20)',
        'vecshift': dist1_vecshift,
        'scalar': dist1_scalar,
    }
]

max_y = max(
    max(dist['vecshift'] + dist['scalar'])
    for dist in distributions
)

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
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylim(0, 450)
    ax.set_ylabel('Throughput (MiB/s)')
    ax.set_title(dist['name'])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_size))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

plt.suptitle('Varint Decoding Throughput by Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('varint_throughput_distributions.svg', format='svg', bbox_inches='tight')
plt.savefig('varint_throughput_distributions.pdf', format='pdf', bbox_inches='tight')
plt.savefig('varint_throughput_distributions.png', format='png', dpi=150, bbox_inches='tight')
print("Saved: varint_throughput_distributions.svg / .pdf / .png")

# Grouped bar charts for the same four distributions
bar_sizes = [1024, 8192, 16384, 32768, 131072, 262144, 524288]
bar_indices = [message_sizes.index(size) for size in bar_sizes]

def select_bar_values(values):
    return [values[i] for i in bar_indices]

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes = axes.flatten()
x = np.arange(len(bar_sizes))
bar_width = 0.32

for idx, dist in enumerate(distributions):
    ax = axes[idx]
    scalar_values = select_bar_values(dist['scalar'])
    vec_values = select_bar_values(dist['vecshift'])

    scalar_bars = ax.bar(
        x - bar_width / 2,
        scalar_values,
        width=bar_width,
        label='Scalar',
        facecolor='white',
        edgecolor='black',
        hatch='///',
        linewidth=0.8,
    )
    vec_bars = ax.bar(
        x + bar_width / 2,
        vec_values,
        width=bar_width,
        label='Rvv',
        facecolor='white',
        edgecolor='black',
        hatch='\\\\',
        linewidth=0.8,
    )

    ax.bar_label(scalar_bars, fmt='%.1f', padding=2, rotation=90, fontsize=8)
    ax.bar_label(vec_bars, fmt='%.1f', padding=2, rotation=90, fontsize=8)

    ax.set_xlabel('Input buffer size')
    ax.set_ylabel('Throughput (MiB/s)')
    ax.set_title(dist['name'])
    ax.set_xticks(x)
    ax.set_xticklabels([format_size(s, None) for s in bar_sizes])
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_ylim(0, 450)
    ax.set_yticks(np.arange(0, 451, 100))
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

plt.suptitle('Varint Decoding Throughput (Grouped Bars)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('varint_throughput_distributions_bar.svg', format='svg', bbox_inches='tight')
plt.savefig('varint_throughput_distributions_bar.pdf', format='pdf', bbox_inches='tight')
plt.savefig('varint_throughput_distributions_bar.png', format='png', dpi=150, bbox_inches='tight')
print("Saved: varint_throughput_distributions_bar.svg / .pdf / .png")
plt.show()
