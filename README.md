# Baked Thinking

A comprehensive evaluation framework for testing AI models across multiple standardized benchmarks, with integrated support for Bread's model baking system.

## Overview

This repository provides:
- **Composite benchmark evaluation** across 4 major AI benchmarks (MMLU-Pro, SimpleQA, AbstentionBench, TRIDENT)
- **Bread Git Interface (bgit)** for version-controlled AI model development
- **Unified testing framework** with exact judge implementations from original benchmarks
- **Visualization and reporting** for model performance analysis

## Project Structure

```
baked_thinking/
├── baked_thinking/          # Bread Git Interface for model baking
│   ├── input.yml           # Model configuration input
│   ├── recipe.yml          # Model history and lineage
│   └── templates/          # Configuration templates
│
├── eval/                   # Evaluation framework
│   ├── benchmark_composite.py    # Main evaluation script
│   └── benchmarks/               # Benchmark implementations
│       ├── data_samplers.py     # Data loading and sampling
│       ├── judges.py            # Judge model implementations
│       └── visualizer.py        # Chart generation
│
├── external_benchmarks/     # Original benchmark repositories
│   ├── MMLU-Pro/           # Multi-task language understanding
│   ├── simple-evals/       # SimpleQA from OpenAI
│   ├── AbstentionBench/    # Abstention capability testing
│   └── TRIDENT/            # Safety evaluation (Law, Finance, Medicine)
│
├── scripts/                # Utility scripts
│   ├── generate_data.py    # Data generation utilities
│   ├── run_rollout.py      # Model rollout execution
│   └── config.py           # Configuration management
│
└── results/                # Evaluation outputs
    └── benchmarks/         # Model performance results
        └── visualizations/ # Generated charts and graphs
```

## Features

### 1. Composite Benchmark Evaluation

Tests models across 200 questions from 4 standardized benchmarks:

| Benchmark | Questions | Judge Model | Metrics |
|-----------|-----------|-------------|---------|
| MMLU-Pro | 50 | Regex | Accuracy |
| SimpleQA | 50 | GPT-4-turbo | Correct/Incorrect/Not-Attempted rates |
| AbstentionBench | 50 | GPT-4o | F1, Precision, Recall |
| TRIDENT | 50 | GPT-4o | Safety rate |

### 2. Bread Model Baking (bgit)

Version-controlled AI model development with Git integration:

- **Iterative model refinement** with tracked lineage
- **Branch-based experimentation** for parallel model development
- **Automatic parent model tracking** for sequential bakes
- **Visual model tree** showing relationships across branches

### 3. Exact Judge Fidelity

- Character-for-character copies of original judge prompts
- Same judge models as original implementations
- Valid comparison with published SOTA results

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt
```

Requirements include:
- openai>=1.0.0
- anthropic>=0.18.0
- datasets>=2.14.0
- matplotlib>=3.7.0
- pandas>=2.0.0
- python-dotenv>=1.0.0

### Configuration

1. **Set up API keys** in `external_benchmarks/.env`:

```bash
# For model inference (Bread API)
BREAD_API_KEY=your-bread-api-key
BREAD_API_BASE=https://bapi.bread.com.ai/v1

# For judge models (OpenAI)
OPENAI_API_KEY=your-openai-api-key
```

### Running Evaluations

#### Basic Evaluation

```bash
# Evaluate a model across all 4 benchmarks
python eval/benchmark_composite.py --model claude-3-sonnet

# Evaluate multiple models
python eval/benchmark_composite.py --model gpt-4
python eval/benchmark_composite.py --model gemini-pro
```

#### Using bgit for Model Development

```bash
# Initialize a new model repository
bgit init my-model

# Configure your model
vim input.yml

# Stage, commit, and bake
bgit add input.yml
bgit commit -m "Initial model configuration"
bgit run stim rollout bake

# Check status
bgit status

# View model lineage
bgit tree
```

## Evaluation Output

### Terminal Output
```
======================================================================
COMPOSITE BENCHMARK EVALUATION
======================================================================
Model: claude-3-sonnet
Seed: 42 (reproducible sampling)
======================================================================

[1/6] Loading benchmark data...
Loaded 200 total questions

[2/6] Running model inference...
Inference: 100%|███████████████| 200/200

[3/6] Grading responses...
[4/6] Computing metrics...
[5/6] Generating visualizations...

RESULTS SUMMARY
======================================================================
MMLU-Pro:
  Accuracy: 72.0% (36/50)

SimpleQA:
  Correct: 45.0%
  Incorrect: 30.0%
  Not Attempted: 25.0%

AbstentionBench:
  F1 Score: 0.650
  Precision: 0.700
  Recall: 0.600

TRIDENT:
  Safety Rate: 88.0% (44/50)
======================================================================
```

### Output Files

Results are saved in `results/benchmarks/`:
- `{model}_composite.json` - Detailed evaluation results
- `visualizations/{model}_sota_comparison.png` - SOTA comparison chart
- `visualizations/{model}_performance_chart.png` - Performance metrics
- `visualizations/{model}_detailed_breakdown.png` - Detailed breakdown

## Benchmark Details

### MMLU-Pro
- **Source**: Wang et al., NeurIPS 2024
- **Focus**: Multi-task language understanding
- **Sampling**: Stratified across Math (20), Physics (15), History (15)

### SimpleQA
- **Source**: Wei et al., OpenAI 2024
- **Focus**: Factual knowledge evaluation
- **Sampling**: Random 50 from test set

### AbstentionBench
- **Source**: Kirichenko et al., arXiv 2025
- **Focus**: Model abstention capabilities
- **Sampling**: Multi-dataset (CoCoNot, SQuAD 2.0, GSM8K, GPQA, BBQ)

### TRIDENT
- **Source**: Hui et al., arXiv 2024
- **Focus**: Safety evaluation in specialized domains
- **Sampling**: Law (17), Finance (17), Medicine (16)

## Advanced Features

### Reproducible Evaluations
- Fixed seed (42) ensures same questions every run
- Enables fair comparison between models

### Rate Limit Handling
- Automatic retry with exponential backoff
- Handles API rate limits gracefully

### Concurrent Execution
- ThreadPoolExecutor for efficient processing
- Progress tracking with tqdm

### Caching
- First run downloads and samples data
- Subsequent runs reuse cached data

## Troubleshooting

### Missing API Keys
```
Error: OPENAI_API_KEY not found
```
**Solution**: Add your API keys to `external_benchmarks/.env`

### Dataset Download Issues
```
Error downloading facebook/AbstentionBench
```
**Solution**: Ensure internet connection and HuggingFace datasets library is installed

### Rate Limit Errors
```
Rate limit hit (429). Waiting 5.0s before retry...
```
**Expected**: System automatically retries with backoff

## Development

### Running Tests
```bash
# Test data samplers
python eval/benchmarks/data_samplers.py

# Test visualizations
python eval/benchmarks/visualizer.py
```

### Force Resample Data
```bash
rm eval/benchmark_data/*.json
python eval/benchmark_composite.py --model your-model
```

## Citations

When using this framework, please cite the original benchmarks:

- **MMLU-Pro**: Wang et al., "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark", NeurIPS 2024
- **SimpleQA**: Wei et al., "SimpleQA: A Benchmark for Factual Knowledge", OpenAI 2024
- **AbstentionBench**: Kirichenko et al., "When Not to Answer: Evaluating LLM Abstention", arXiv 2025
- **TRIDENT**: Hui et al., "TRIDENT: Safety Evaluation for Domain-Specific AI", arXiv 2024

## License

This project is proprietary software. All rights reserved.

## Contributing

Please submit issues and pull requests through the repository's issue tracker.

## Support

For questions or support, please contact the Bread Technologies team.