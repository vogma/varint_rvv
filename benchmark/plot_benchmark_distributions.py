#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import defaultdict
from math import ceil

import matplotlib.pyplot as plt
import numpy as np


def parse_benchmark_name(name):
    """
    Parse benchmark name like 'BM<varint_rvv, 95, 2, 1, 1, 1>/1024'
    Returns: (algorithm, distribution_tuple, size) or None if parsing fails
    """
    pattern = r'BM<([^,]+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)>/(\d+)'
    match = re.match(pattern, name)
    if not match:
        return None
    algorithm = match.group(1)
    distribution = tuple(int(match.group(i)) for i in range(2, 7))
    size = int(match.group(7))
    return algorithm, distribution, size


def distribution_label(dist_tuple):
    """Create human-readable label for distribution tuple."""
    dist_str = '-'.join(str(x) for x in dist_tuple)
    # Classify distribution type
    if dist_tuple[0] >= 90:
        desc = "Heavily 1-byte"
    elif dist_tuple[0] >= 80:
        desc = "Mostly 1-byte"
    elif dist_tuple[0] >= 70:
        desc = "Skewed 1-byte"
    elif all(x == dist_tuple[0] for x in dist_tuple):
        desc = "Uniform"
    else:
        desc = "Mixed"
    return f"{desc} ({dist_str})"


def load_benchmark_data(json_path):
    """
    Load and parse benchmark JSON file.
    Returns: (context, data_by_distribution)
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)

    context = raw.get('context', {})
    benchmarks = raw.get('benchmarks', [])

    # Organize: {distribution: {algorithm: {'sizes': [], 'throughput': []}}}
    data = defaultdict(lambda: defaultdict(lambda: {'sizes': [], 'throughput': [], 'bytes_cycle': []}))

    for bm in benchmarks:
        parsed = parse_benchmark_name(bm['name'])
        if not parsed:
            continue
        algorithm, distribution, size = parsed

        # Convert bytes/second to MiB/s
        throughput_mibs = bm['bytes_per_second'] / (1024 * 1024)
        bytes_cycle = bm.get('bytes/cycle', 0)

        data[distribution][algorithm]['sizes'].append(size)
        data[distribution][algorithm]['throughput'].append(throughput_mibs)
        data[distribution][algorithm]['bytes_cycle'].append(bytes_cycle)

    # Sort each algorithm's data by size
    for dist in data:
        for algo in data[dist]:
            sorted_pairs = sorted(zip(
                data[dist][algo]['sizes'],
                data[dist][algo]['throughput'],
                data[dist][algo]['bytes_cycle']
            ))
            data[dist][algo]['sizes'] = [p[0] for p in sorted_pairs]
            data[dist][algo]['throughput'] = [p[1] for p in sorted_pairs]
            data[dist][algo]['bytes_cycle'] = [p[2] for p in sorted_pairs]

    return context, dict(data)


def get_algorithms(data):
    """Get sorted list of all unique algorithms across distributions."""
    algos = set()
    for dist in data.values():
        algos.update(dist.keys())
    return sorted(algos)


# Color palette for algorithms
ALGO_COLORS = ['#2563eb', '#dc2626', '#059669', '#7c3aed', '#ea580c', '#0891b2']
ALGO_MARKERS = ['o', 's', '^', 'D', 'v', 'p']

def format_size(x, pos=None):
    if x >= 1048576:
        return f'{int(x/1048576)}M'
    elif x >= 1024:
        return f'{int(x/1024)}K'
    return str(int(x))


def plot_line_charts(data, context, output_prefix):
    """Generate line chart subplots for each distribution."""
    distributions = sorted(data.keys(), key=lambda d: -d[0])  # Sort by first percentage descending
    algorithms = get_algorithms(data)

    n_dist = len(distributions)
    if n_dist == 0:
        print("No data to plot")
        return

    cols = 2
    rows = ceil(n_dist / cols)

    # Calculate global max for consistent y-axis
    max_throughput = 0
    for dist_data in data.values():
        for algo_data in dist_data.values():
            if algo_data['throughput']:
                max_throughput = max(max_throughput, max(algo_data['throughput']))
    y_max = ceil(max_throughput / 50) * 50 + 50  # Round up to nearest 50

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.5 * rows))
    if n_dist == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, dist in enumerate(distributions):
        ax = axes[idx]
        dist_data = data[dist]

        for algo_idx, algo in enumerate(algorithms):
            if algo not in dist_data:
                continue
            algo_data = dist_data[algo]
            color = ALGO_COLORS[algo_idx % len(ALGO_COLORS)]
            marker = ALGO_MARKERS[algo_idx % len(ALGO_MARKERS)]

            ax.plot(algo_data['sizes'], algo_data['throughput'],
                    f'{marker}-', label=algo, color=color, linewidth=1.5, markersize=4)

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Message Size (bytes)')
        ax.set_ylabel('Throughput (MiB/s)')
        ax.set_ylim(0, y_max)
        ax.set_title(distribution_label(dist))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_size))
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for idx in range(n_dist, rows * cols):
        axes[idx].set_visible(False)

    # Add context info
    host = context.get('host_name', 'unknown')
    date = context.get('date', '')[:10]
    suptitle = f'Varint Decoding Throughput by Distribution\n({host}, {date})'
    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()

    for fmt in ['svg', 'pdf', 'png']:
        filename = f'{output_prefix}_line.{fmt}'
        kwargs = {'dpi': 150} if fmt == 'png' else {}
        plt.savefig(filename, format=fmt, bbox_inches='tight', **kwargs)
    print(f"Saved: {output_prefix}_line.svg / .pdf / .png")
    plt.close(fig)


def plot_bar_charts(data, context, output_prefix):
    """Generate grouped bar chart subplots for each distribution."""
    distributions = sorted(data.keys(), key=lambda d: -d[0])
    algorithms = get_algorithms(data)

    n_dist = len(distributions)
    n_algo = len(algorithms)
    if n_dist == 0 or n_algo == 0:
        return

    cols = 2
    rows = ceil(n_dist / cols)

    # Fixed bar sizes: 256, 512, 1K, 4K, 8K, 64K, 256K, 1M
    bar_sizes = [256, 512, 1024, 4096, 8192, 65536, 262144, 1048576]

    # Calculate global max
    max_throughput = 0
    for dist_data in data.values():
        for algo_data in dist_data.values():
            if algo_data['throughput']:
                max_throughput = max(max_throughput, max(algo_data['throughput']))
    y_max = ceil(max_throughput / 50) * 50 + 50

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if n_dist == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    x = np.arange(len(bar_sizes))
    total_width = 0.7
    bar_width = total_width / n_algo

    # Different shading styles for each algorithm
    bar_styles = [
        {'facecolor': '#d1d5db', 'hatch': '', 'edgecolor': '#374151'},       # solid gray
        {'facecolor': 'white', 'hatch': '//\\\\', 'edgecolor': '#374151'},   # white with diagonal grid
        {'facecolor': '#fecaca', 'hatch': '', 'edgecolor': '#991b1b'},       # light red solid
        {'facecolor': 'white', 'hatch': '...', 'edgecolor': '#166534'},      # white with dots
        {'facecolor': '#e9d5ff', 'hatch': '|||', 'edgecolor': '#6b21a8'},    # light purple with vertical
        {'facecolor': '#fed7aa', 'hatch': '---', 'edgecolor': '#c2410c'},    # light orange with horizontal
    ]

    for idx, dist in enumerate(distributions):
        ax = axes[idx]
        dist_data = data[dist]

        for algo_idx, algo in enumerate(algorithms):
            if algo not in dist_data:
                continue
            algo_data = dist_data[algo]

            # Get values for bar_sizes
            size_to_val = dict(zip(algo_data['sizes'], algo_data['throughput']))
            values = [size_to_val.get(s, 0) for s in bar_sizes]

            style = bar_styles[algo_idx % len(bar_styles)]
            offset = (algo_idx - (n_algo - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=algo,
                facecolor=style['facecolor'],
                edgecolor=style['edgecolor'],
                hatch=style['hatch'],
                linewidth=1.0,
            )
            ax.bar_label(bars, fmt='%.0f', padding=2, rotation=90, fontsize=6)

        ax.set_xlabel('Input buffer size')
        ax.set_ylabel('Throughput (MiB/s)')
        ax.set_title(distribution_label(dist))
        ax.set_xticks(x)
        ax.set_xticklabels([format_size(s) for s in bar_sizes])
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_ylim(0, y_max)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

    # Hide unused subplots
    for idx in range(n_dist, rows * cols):
        axes[idx].set_visible(False)

    host = context.get('host_name', 'unknown')
    date = context.get('date', '')[:10]
    suptitle = f'Varint Decoding Throughput (Grouped Bars)\n({host}, {date})'
    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()

    for fmt in ['svg', 'pdf', 'png']:
        filename = f'{output_prefix}_bar.{fmt}'
        kwargs = {'dpi': 150} if fmt == 'png' else {}
        plt.savefig(filename, format=fmt, bbox_inches='tight', **kwargs)
    print(f"Saved: {output_prefix}_bar.svg / .pdf / .png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot varint benchmark results from Google Benchmark JSON output'
    )
    parser.add_argument(
        '-i', '--input',
        default='results.json',
        help='Input JSON file from benchmark (default: results.json)'
    )
    parser.add_argument(
        '-o', '--output',
        default='varint_throughput_distributions',
        help='Output filename prefix (default: varint_throughput_distributions)'
    )
    parser.add_argument(
        '--no-bar',
        action='store_true',
        help='Skip bar chart generation'
    )
    parser.add_argument(
        '--no-line',
        action='store_true',
        help='Skip line chart generation'
    )
    args = parser.parse_args()

    print(f"Loading benchmark data from: {args.input}")
    try:
        context, data = load_benchmark_data(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print(f"Host: {context.get('host_name', 'unknown')}")
    print(f"Date: {context.get('date', 'unknown')}")
    print(f"Found {len(data)} distribution(s):")
    for dist in sorted(data.keys(), key=lambda d: -d[0]):
        algos = list(data[dist].keys())
        print(f"  {distribution_label(dist)}: {', '.join(algos)}")

    if not args.no_line:
        plot_line_charts(data, context, args.output)

    if not args.no_bar:
        plot_bar_charts(data, context, args.output)

    print("Done.")


if __name__ == '__main__':
    main()
