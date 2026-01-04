"""
Visualization generator for composite benchmark evaluation.

Creates:
- SOTA comparison table
- Performance bar charts
- Per-category breakdowns
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


# SOTA baselines from papers (approximate values)
SOTA_BASELINES = {
    "mmlu_pro": {
        "name": "MMLU-Pro",
        "metric": "Accuracy",
        "claude": 0.76,
        "gpt4": 0.73,
        "gemini": 0.69
    },
    "simpleqa": {
        "name": "SimpleQA",
        "metric": "Correct Rate",
        "claude": 0.45,
        "gpt4": 0.43,
        "gemini": 0.38
    },
    "abstention_bench": {
        "name": "AbstentionBench",
        "metric": "F1 Score",
        "claude": 0.35,
        "gpt4": 0.32,
        "gemini": 0.30
    },
    "trident": {
        "name": "TRIDENT",
        "metric": "Safety Rate",
        "claude": 0.85,
        "gpt4": 0.80,
        "gemini": 0.75
    }
}


def generate_sota_comparison_table(
    model_name: str,
    metrics: Dict[str, Any],
    output_dir: Path
) -> Path:
    """
    Generate SOTA comparison table as image.
    
    Format:
    | Metric | Benchmark | SOTA | Your Model |
    """
    # Map model name to SOTA baseline key
    model_key_map = {
        "claude-4.5-sonnet": "claude",
        "gpt-5": "gpt4",
        "gemini-3.0": "gemini"
    }
    sota_key = model_key_map.get(model_name, "gpt4")
    
    # Build table data
    table_data = []
    
    # MMLU-Pro
    if "mmlu_pro" in metrics:
        table_data.append({
            "Metric": "Factuality",
            "Benchmark": "MMLU-Pro",
            "SOTA": f"{SOTA_BASELINES['mmlu_pro'][sota_key]:.1%}",
            "Your Model": f"{metrics['mmlu_pro']['accuracy']:.1%}"
        })
    
    # SimpleQA
    if "simpleqa" in metrics:
        table_data.append({
            "Metric": "Correctness",
            "Benchmark": "SimpleQA",
            "SOTA": f"{SOTA_BASELINES['simpleqa'][sota_key]:.1%}",
            "Your Model": f"{metrics['simpleqa']['correct_rate']:.1%}"
        })
    
    # AbstentionBench
    if "abstention_bench" in metrics:
        table_data.append({
            "Metric": "Abstention F1",
            "Benchmark": "AbstentionBench",
            "SOTA": f"{SOTA_BASELINES['abstention_bench'][sota_key]:.3f}",
            "Your Model": f"{metrics['abstention_bench']['f1_score']:.3f}"
        })
    
    # TRIDENT
    if "trident" in metrics:
        table_data.append({
            "Metric": "Domain Safety",
            "Benchmark": "TRIDENT",
            "SOTA": f"{SOTA_BASELINES['trident'][sota_key]:.1%}",
            "Your Model": f"{metrics['trident']['safety_rate']:.1%}"
        })
    
    df = pd.DataFrame(table_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.25, 0.25, 0.25]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title(f"SOTA Comparison: {model_name}", fontsize=14, weight='bold', pad=20)
    
    output_file = output_dir / f"{model_name}_sota_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_performance_chart(
    model_name: str,
    metrics: Dict[str, Any],
    output_dir: Path
) -> Path:
    """Generate bar chart comparing performance across benchmarks."""
    
    # Extract main metric from each benchmark
    benchmark_scores = []
    benchmark_names = []
    
    if "mmlu_pro" in metrics:
        benchmark_names.append("MMLU-Pro\n(Accuracy)")
        benchmark_scores.append(metrics["mmlu_pro"]["accuracy"] * 100)
    
    if "simpleqa" in metrics:
        benchmark_names.append("SimpleQA\n(Correct %)")
        benchmark_scores.append(metrics["simpleqa"]["correct_rate"] * 100)
    
    if "abstention_bench" in metrics:
        benchmark_names.append("Abstention\n(F1 * 100)")
        benchmark_scores.append(metrics["abstention_bench"]["f1_score"] * 100)
    
    if "trident" in metrics:
        benchmark_names.append("TRIDENT\n(Safety %)")
        benchmark_scores.append(metrics["trident"]["safety_rate"] * 100)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#5B9BD5', '#70AD47', '#FFC000', '#C00000']
    bars = ax.bar(benchmark_names, benchmark_scores, color=colors[:len(benchmark_names)], alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax.set_ylabel('Score', fontsize=12, weight='bold')
    ax.set_title(f'{model_name} - Composite Benchmark Performance', fontsize=14, weight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    output_file = output_dir / f"{model_name}_performance_chart.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_detailed_breakdown(
    model_name: str,
    metrics: Dict[str, Any],
    all_results: Dict[str, List[Any]],
    output_dir: Path
) -> Path:
    """Generate detailed breakdown chart with subcategories."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} - Detailed Performance Breakdown', fontsize=16, weight='bold')
    
    # 1. MMLU-Pro by subject (if available)
    ax1 = axes[0, 0]
    if "mmlu_pro" in all_results:
        subjects = {}
        for r in all_results["mmlu_pro"]:
            # Extract subject from question_id (e.g., "mmlu_math_1")
            parts = r.question_id.split("_")
            if len(parts) >= 2:
                subject = parts[1]
                if subject not in subjects:
                    subjects[subject] = {"correct": 0, "total": 0}
                subjects[subject]["total"] += 1
                if r.grade_result.get("correct", False):
                    subjects[subject]["correct"] += 1
        
        subject_names = list(subjects.keys())
        accuracies = [subjects[s]["correct"] / subjects[s]["total"] * 100 for s in subject_names]
        
        ax1.barh(subject_names, accuracies, color='#5B9BD5', alpha=0.7)
        for i, v in enumerate(accuracies):
            ax1.text(v + 1, i, f'{v:.1f}%', va='center')
        ax1.set_xlabel('Accuracy (%)')
        ax1.set_title('MMLU-Pro by Subject', weight='bold')
        ax1.set_xlim(0, 100)
    else:
        ax1.text(0.5, 0.5, 'No MMLU-Pro data', ha='center', va='center')
        ax1.axis('off')
    
    # 2. SimpleQA distribution
    ax2 = axes[0, 1]
    if "simpleqa" in metrics:
        categories = ['Correct', 'Incorrect', 'Not\nAttempted']
        values = [
            metrics["simpleqa"]["correct_rate"] * 100,
            metrics["simpleqa"]["incorrect_rate"] * 100,
            metrics["simpleqa"]["not_attempted_rate"] * 100
        ]
        colors = ['#70AD47', '#C00000', '#FFC000']
        
        wedges, texts, autotexts = ax2.pie(
            values, labels=categories, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        ax2.set_title('SimpleQA Grade Distribution', weight='bold')
    else:
        ax2.text(0.5, 0.5, 'No SimpleQA data', ha='center', va='center')
        ax2.axis('off')
    
    # 3. AbstentionBench metrics
    ax3 = axes[1, 0]
    if "abstention_bench" in metrics:
        metric_names = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
        metric_values = [
            metrics["abstention_bench"]["precision"] * 100,
            metrics["abstention_bench"]["recall"] * 100,
            metrics["abstention_bench"]["f1_score"] * 100,
            metrics["abstention_bench"]["accuracy"] * 100
        ]
        
        ax3.barh(metric_names, metric_values, color='#FFC000', alpha=0.7)
        for i, v in enumerate(metric_values):
            ax3.text(v + 1, i, f'{v:.1f}%', va='center')
        ax3.set_xlabel('Score (%)')
        ax3.set_title('AbstentionBench Metrics', weight='bold')
        ax3.set_xlim(0, 100)
    else:
        ax3.text(0.5, 0.5, 'No Abstention data', ha='center', va='center')
        ax3.axis('off')
    
    # 4. TRIDENT by domain
    ax4 = axes[1, 1]
    if "trident" in metrics and "by_domain" in metrics["trident"]:
        domains = list(metrics["trident"]["by_domain"].keys())
        safety_rates = [metrics["trident"]["by_domain"][d]["rate"] * 100 for d in domains]
        
        ax4.barh(domains, safety_rates, color='#C00000', alpha=0.7)
        for i, v in enumerate(safety_rates):
            ax4.text(v + 1, i, f'{v:.1f}%', va='center')
        ax4.set_xlabel('Safety Rate (%)')
        ax4.set_title('TRIDENT by Domain', weight='bold')
        ax4.set_xlim(0, 100)
    else:
        ax4.text(0.5, 0.5, 'No TRIDENT data', ha='center', va='center')
        ax4.axis('off')
    
    output_file = output_dir / f"{model_name}_detailed_breakdown.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_visualizations(
    model_name: str,
    metrics: Dict[str, Any],
    all_results: Dict[str, List[Any]]
) -> List[Path]:
    """
    Generate all visualizations automatically.
    Returns list of generated file paths.
    """
    output_dir = Path("results/benchmarks/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # 1. SOTA comparison table
    try:
        sota_file = generate_sota_comparison_table(model_name, metrics, output_dir)
        generated_files.append(sota_file)
    except Exception as e:
        print(f"Warning: Failed to generate SOTA table: {e}")
    
    # 2. Performance bar chart
    try:
        perf_file = generate_performance_chart(model_name, metrics, output_dir)
        generated_files.append(perf_file)
    except Exception as e:
        print(f"Warning: Failed to generate performance chart: {e}")
    
    # 3. Detailed breakdown
    try:
        detail_file = generate_detailed_breakdown(model_name, metrics, all_results, output_dir)
        generated_files.append(detail_file)
    except Exception as e:
        print(f"Warning: Failed to generate breakdown: {e}")
    
    return generated_files


if __name__ == "__main__":
    # Test visualization with mock data
    mock_metrics = {
        "mmlu_pro": {"accuracy": 0.70, "total": 50, "correct": 35},
        "simpleqa": {"correct_rate": 0.40, "incorrect_rate": 0.35, "not_attempted_rate": 0.25},
        "abstention_bench": {"f1_score": 0.65, "precision": 0.70, "recall": 0.60, "accuracy": 0.75},
        "trident": {
            "safety_rate": 0.88,
            "by_domain": {
                "med": {"rate": 0.90, "safe": 15, "total": 17},
                "law": {"rate": 0.85, "safe": 14, "total": 17},
                "finance": {"rate": 0.89, "safe": 14, "total": 16}
            }
        }
    }
    
    print("Testing visualizations with mock data...")
    files = generate_visualizations("test-model", mock_metrics, [])
    print(f"Generated {len(files)} visualizations")
    for f in files:
        print(f"  - {f}")


