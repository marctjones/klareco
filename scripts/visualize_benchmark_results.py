#!/usr/bin/env python3
"""
Visualize benchmark results from benchmark_slot_retrievers.py.

Creates comparison charts for latency, memory, and accuracy.

Usage:
    python scripts/visualize_benchmark_results.py \
        --results benchmark_results.json \
        --output benchmark_report.html
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


def plot_latency_comparison(solutions, output_path):
    """Create latency comparison bar chart."""
    names = [s['retriever'] for s in solutions if 'error' not in s]
    mean_latencies = [s['latency']['mean_ms'] for s in solutions if 'error' not in s]
    p95_latencies = [s['latency']['p95_ms'] for s in solutions if 'error' not in s]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(names))
    width = 0.35

    ax.bar([i - width/2 for i in x], mean_latencies, width, label='Mean', alpha=0.8)
    ax.bar([i + width/2 for i in x], p95_latencies, width, label='P95', alpha=0.8)

    ax.set_xlabel('Solution')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Query Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved latency chart: {output_path}")


def plot_memory_comparison(solutions, output_path):
    """Create memory usage comparison bar chart."""
    names = [s['retriever'] for s in solutions if 'error' not in s]
    peak_memory = [s['memory']['peak_mb'] for s in solutions if 'error' not in s]
    delta_memory = [s['memory']['delta_mb'] for s in solutions if 'error' not in s]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(names))
    width = 0.35

    ax.bar([i - width/2 for i in x], peak_memory, width, label='Peak', alpha=0.8)
    ax.bar([i + width/2 for i in x], delta_memory, width, label='Delta', alpha=0.8)

    ax.set_xlabel('Solution')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved memory chart: {output_path}")


def plot_accuracy_comparison(solutions, output_path):
    """Create accuracy metrics comparison bar chart."""
    names = [s['retriever'] for s in solutions if 'error' not in s and 'accuracy' in s]
    if not names:
        print("  Skipping accuracy chart: no accuracy data")
        return

    recall = [s['accuracy']['recall_at_10'] for s in solutions if 'error' not in s and 'accuracy' in s]
    mrr = [s['accuracy']['mrr'] for s in solutions if 'error' not in s and 'accuracy' in s]
    ndcg = [s['accuracy']['ndcg_at_10'] for s in solutions if 'error' not in s and 'accuracy' in s]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(names))
    width = 0.25

    ax.bar([i - width for i in x], recall, width, label='Recall@10', alpha=0.8)
    ax.bar(x, mrr, width, label='MRR', alpha=0.8)
    ax.bar([i + width for i in x], ndcg, width, label='NDCG@10', alpha=0.8)

    ax.set_xlabel('Solution')
    ax.set_ylabel('Score')
    ax.set_title('Accuracy Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved accuracy chart: {output_path}")


def plot_tradeoff_scatter(solutions, output_path):
    """Create latency vs memory tradeoff scatter plot."""
    names = [s['retriever'] for s in solutions if 'error' not in s]
    latencies = [s['latency']['mean_ms'] for s in solutions if 'error' not in s]
    memories = [s['memory']['delta_mb'] for s in solutions if 'error' not in s]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(range(len(names)))

    for i, (name, lat, mem) in enumerate(zip(names, latencies, memories)):
        ax.scatter(lat, mem, s=200, alpha=0.6, c=[colors[i]], label=name)
        ax.annotate(name, (lat, mem), xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Mean Latency (ms)')
    ax.set_ylabel('Memory Delta (MB)')
    ax.set_title('Speed vs Memory Tradeoff')
    ax.grid(alpha=0.3)

    # Draw quadrant lines at median
    if latencies and memories:
        med_lat = sorted(latencies)[len(latencies)//2]
        med_mem = sorted(memories)[len(memories)//2]
        ax.axvline(med_lat, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(med_mem, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved tradeoff chart: {output_path}")


def generate_html_report(results, output_path, charts_dir):
    """Generate HTML report with embedded charts."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Slot Retriever Benchmark Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2 {{
            color: #333;
        }}
        .metadata {{
            background: white;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .winner {{
            background-color: #c8e6c9;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <h1>Slot-Based Retriever Benchmark Results</h1>

    <div class="metadata">
        <p><strong>Index:</strong> {results['index_path']}</p>
        <p><strong>Queries:</strong> {results['queries_path']}</p>
        <p><strong>Number of Queries:</strong> {results['num_queries']}</p>
    </div>

    <h2>Summary Table</h2>
    <table>
        <tr>
            <th>Solution</th>
            <th>Mean Latency (ms)</th>
            <th>P95 Latency (ms)</th>
            <th>Memory Delta (MB)</th>
            <th>Recall@10</th>
            <th>MRR</th>
        </tr>
"""

    # Find winners
    valid_solutions = [s for s in results['solutions'] if 'error' not in s]

    if valid_solutions:
        fastest = min(valid_solutions, key=lambda s: s['latency']['mean_ms'])
        lowest_mem = min(valid_solutions, key=lambda s: s['memory']['delta_mb'])
        solutions_with_accuracy = [s for s in valid_solutions if 'accuracy' in s]
        best_recall = max(solutions_with_accuracy, key=lambda s: s['accuracy']['recall_at_10']) if solutions_with_accuracy else None

        for sol in results['solutions']:
            if 'error' in sol:
                html += f"""
        <tr>
            <td>{sol['retriever']}</td>
            <td colspan="5">ERROR: {sol['error']}</td>
        </tr>
"""
                continue

            name = sol['retriever']
            mean_lat = sol['latency']['mean_ms']
            p95_lat = sol['latency']['p95_ms']
            mem = sol['memory']['delta_mb']
            recall = sol.get('accuracy', {}).get('recall_at_10', 0.0)
            mrr = sol.get('accuracy', {}).get('mrr', 0.0)

            fastest_class = ' class="winner"' if sol == fastest else ''
            mem_class = ' class="winner"' if sol == lowest_mem else ''
            recall_class = ' class="winner"' if best_recall and sol == best_recall else ''

            html += f"""
        <tr>
            <td>{name}</td>
            <td{fastest_class}>{mean_lat:.2f}</td>
            <td>{p95_lat:.2f}</td>
            <td{mem_class}>{mem:.1f}</td>
            <td{recall_class}>{recall:.3f}</td>
            <td>{mrr:.3f}</td>
        </tr>
"""

    html += """
    </table>

    <h2>Charts</h2>
"""

    # Add charts
    chart_files = [
        ('latency_comparison.png', 'Latency Comparison'),
        ('memory_comparison.png', 'Memory Comparison'),
        ('accuracy_comparison.png', 'Accuracy Comparison'),
        ('tradeoff_scatter.png', 'Speed vs Memory Tradeoff'),
    ]

    for chart_file, title in chart_files:
        chart_path = charts_dir / chart_file
        if chart_path.exists():
            html += f"""
    <div class="chart">
        <h3>{title}</h3>
        <img src="{chart_file}" alt="{title}">
    </div>
"""

    html += """
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"  Saved HTML report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to benchmark results JSON'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmark_report.html'),
        help='Output path for HTML report'
    )

    args = parser.parse_args()

    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        sys.exit(1)

    # Load results
    with open(args.results) as f:
        results = json.load(f)

    print(f"Loaded results: {len(results['solutions'])} solutions")

    # Create output directory for charts
    charts_dir = args.output.parent / "charts"
    charts_dir.mkdir(exist_ok=True)

    # Generate charts
    print("Generating charts...")
    plot_latency_comparison(results['solutions'], charts_dir / 'latency_comparison.png')
    plot_memory_comparison(results['solutions'], charts_dir / 'memory_comparison.png')
    plot_accuracy_comparison(results['solutions'], charts_dir / 'accuracy_comparison.png')
    plot_tradeoff_scatter(results['solutions'], charts_dir / 'tradeoff_scatter.png')

    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(results, args.output, charts_dir)

    print()
    print(f"Done! Open {args.output} in a browser to view the report.")


if __name__ == '__main__':
    main()
