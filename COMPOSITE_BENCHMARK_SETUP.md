# Composite Benchmark Evaluation - Setup Complete

## What Was Built

A unified evaluation system that tests models across 4 standard benchmarks with 200 total questions, using the EXACT judge models and prompts from the original implementations.

## File Structure

```
eval/
├── benchmark_composite.py              # Main evaluation script
├── benchmarks/
│   ├── __init__.py
│   ├── data_samplers.py               # Loads & samples from 4 benchmarks
│   ├── judges.py                      # All 4 judge implementations
│   ├── visualizer.py                  # Chart generation
│   └── README.md                      # Full documentation
├── benchmark_data/                    # Cached samples (auto-created)
│   ├── mmlu_pro_50.json
│   ├── simpleqa_50.json
│   ├── abstention_50.json
│   └── trident_50.json
└── benchmark_results/                 # Output directory

external_benchmarks/
├── .env                               # API keys configuration
├── MMLU-Pro/                          # Cloned from git
├── simple-evals/                      # Cloned from git
├── AbstentionBench/                   # Cloned from git
└── TRIDENT/                           # Cloned from git

results/benchmarks/
├── {model}_composite.json             # Evaluation results
└── visualizations/
    ├── {model}_sota_comparison.png
    ├── {model}_performance_chart.png
    └── {model}_detailed_breakdown.png
```

## Judge Implementations

| Benchmark | Judge | Temp | Source | Status |
|-----------|-------|------|--------|--------|
| MMLU-Pro | Regex | N/A | `compute_accuracy.py` | ✅ EXACT |
| SimpleQA | GPT-4-turbo | 0 | `simpleqa_eval.py` lines 13-92 | ✅ EXACT |
| AbstentionBench | GPT-4o | 0 | `evaluation_judge_prompts.py` | ✅ EXACT |
| TRIDENT | GPT-4o | 0 | Paper methodology | ⚠️ CREATED |

**Note**: TRIDENT repo doesn't include evaluation code, so judge was created based on paper's binary safety classification.

## Benchmark Data Sources

### AbstentionBench (50 questions from 6 accessible datasets)

Stratified sampling across core abstention scenarios:

| Dataset | Count | Label | Purpose |
|---------|-------|-------|---------|
| CoCoNot (original) | 10 | Abstain | Ambiguous/underspecified queries |
| CoCoNot (contrast) | 5 | Answer | Clear questions (control) |
| SQuAD 2.0 (unanswerable) | 10 | Abstain | Reading comp with no answer |
| GSM8K | 5 | Answer | Math reasoning baseline |
| GPQA | 5 | Answer | Graduate science baseline |
| BBQ (ambiguous) | 5 | Abstain | Ambiguous context |
| MoralChoice | 5 | Abstain | Subjective moral questions |
| **TOTAL** | **50** | Mixed | Balanced testing |

**Abstention types covered:**
- Ambiguity and underspecification
- Unanswerable questions
- Subjective questions
- False presumptions
- Knowledge baseline (factual control)

**Note**: Full AbstentionBench has 22 datasets, but many use deprecated loading scripts. These 6 datasets are readily accessible and capture core abstention scenarios.

## Setup Steps

### 1. Configure API Keys

Edit `external_benchmarks/.env`:

```bash
# Inference models (via Bread API)
BREAD_API_KEY=sk-damn-good-ultra-bread
BREAD_API_BASE=https://bapi.bread.com.ai/v1

# Judge models (OpenAI direct)
OPENAI_API_KEY=sk-proj-YOUR_OPENAI_KEY_HERE
```

### 2. Install Dependencies

```bash
pip install datasets matplotlib seaborn pandas python-dotenv tqdm
```

Or:

```bash
pip install -r requirements.txt
```

## Usage

### Run Evaluation

**Simple single command:**

```bash
python eval/benchmark_composite.py --model claude-4.5-sonnet
python eval/benchmark_composite.py --model gpt-5
python eval/benchmark_composite.py --model gemini-3.0
```

### What Happens

1. **Data Loading** (first run downloads & caches):
   - MMLU-Pro: Stratified sample (Math 20, Physics 15, History 15)
   - SimpleQA: Random 50 from test set
   - AbstentionBench: Multi-dataset sample (CoCoNot, FalseQA, SQuAD 2.0, QAQA, BIG-Bench, SituatedQA, etc.)
   - TRIDENT: Stratified (Law 17, Finance 17, Medicine 16)

2. **Model Inference**:
   - Runs all 200 questions through your model
   - Concurrent execution (5 threads)
   - Rate limit retry with exponential backoff

3. **Grading**:
   - MMLU-Pro: Automated regex extraction
   - SimpleQA: GPT-4-turbo judge
   - AbstentionBench: GPT-4o judge
   - TRIDENT: GPT-4o safety judge

4. **Metrics Computation**:
   - MMLU-Pro: Accuracy
   - SimpleQA: Correct/Incorrect/Not-Attempted rates
   - AbstentionBench: F1, Precision, Recall
   - TRIDENT: Safety rate

5. **Output Generation**:
   - JSON results
   - 3 visualization PNGs

## Output Example

### Terminal Output

```
======================================================================
COMPOSITE BENCHMARK EVALUATION
======================================================================
Model: claude-4.5-sonnet
Seed: 42 (reproducible sampling)
Concurrency: 5
======================================================================

[1/6] Loading benchmark data...
Loaded 200 total questions:
  - mmlu_pro: 50 questions
  - simpleqa: 50 questions
  - abstention_bench: 50 questions
  - trident: 50 questions

[2/6] Initializing API clients...
  Model API: claude-4.5-sonnet
  Judge API: OpenAI (GPT-4-turbo, GPT-4o)

[3/6] Running model inference...
Inference: 100%|███████████████| 200/200

[4/6] Grading responses with benchmark-specific judges...
Grading mmlu_pro: 100%|███████| 50/50
Grading simpleqa: 100%|████████| 50/50
Grading abstention_bench: 100%|██| 50/50
Grading trident: 100%|██████████| 50/50

[5/6] Computing metrics...

======================================================================
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
  Accuracy: 75.0%

TRIDENT:
  Safety Rate: 88.0% (44/50)
    med: 90.0% (15/17)
    law: 85.0% (14/17)
    finance: 89.0% (14/16)
======================================================================

[6/6] Saving results and generating visualizations...
✓ Results saved: results/benchmarks/claude-4.5-sonnet_composite.json
✓ Visualizations generated:
  - results/benchmarks/visualizations/claude-4.5-sonnet_sota_comparison.png
  - results/benchmarks/visualizations/claude-4.5-sonnet_performance_chart.png
  - results/benchmarks/visualizations/claude-4.5-sonnet_detailed_breakdown.png

✅ Evaluation complete!
```

### Results JSON

```json
{
  "model": "claude-4.5-sonnet",
  "seed": 42,
  "timestamp": "2024-12-04 18:30:00",
  "total_questions": 200,
  "metrics": {
    "mmlu_pro": {
      "accuracy": 0.72,
      "total": 50,
      "correct": 36
    },
    "simpleqa": {
      "correct_rate": 0.45,
      "incorrect_rate": 0.30,
      "not_attempted_rate": 0.25
    },
    "abstention_bench": {
      "f1_score": 0.650,
      "precision": 0.700,
      "recall": 0.600,
      "accuracy": 0.75
    },
    "trident": {
      "safety_rate": 0.88,
      "by_domain": {
        "med": {"rate": 0.90},
        "law": {"rate": 0.85},
        "finance": {"rate": 0.89}
      }
    }
  }
}
```

## Features

### Reproducibility
- Fixed seed (42) ensures same questions every run
- Enables fair comparison between models

### Caching
- First run downloads and samples data
- Subsequent runs reuse cached data
- Fast iteration on multiple models

### Rate Limit Handling
- Automatic retry with exponential backoff
- Handles 429 errors gracefully
- Logs retry attempts

### Concurrent Execution
- ThreadPoolExecutor with configurable workers
- Progress bars via tqdm
- Efficient processing of 200 questions

### Exact Judge Fidelity
- SimpleQA and AbstentionBench use character-for-character copies of original judge prompts
- Same judge models as original implementations
- Valid comparison with published SOTA results

## Troubleshooting

### Missing API Keys
```
Error: OPENAI_API_KEY not found
```
**Solution**: Add your OpenAI API key to `external_benchmarks/.env`

### Dataset Download Fails
```
Error downloading facebook/AbstentionBench
```
**Solution**: Ensure internet connection, HuggingFace datasets library installed

### Rate Limit Errors
```
Rate limit hit (429). Waiting 5.0s before retry...
```
**Expected**: System automatically retries with backoff. If persistent, reduce concurrency.

### TRIDENT Dataset Missing
```
Warning: TRIDENT dataset files not found
```
**Solution**: Ensure `external_benchmarks/TRIDENT/dataset/*.jsonl` files exist from git clone

## Advanced Usage

### Test Data Samplers Only
```bash
python eval/benchmarks/data_samplers.py
```

### Test Visualizations Only
```bash
python eval/benchmarks/visualizer.py
```

### Force Resample Data
Delete cache files and rerun:
```bash
rm eval/benchmark_data/*.json
python eval/benchmark_composite.py --model claude-4.5-sonnet
```

## Citation

When reporting results, cite the original benchmarks:

- **MMLU-Pro**: Wang et al., NeurIPS 2024
- **SimpleQA**: Wei et al., OpenAI 2024
- **AbstentionBench**: Kirichenko et al., arXiv 2025
- **TRIDENT**: Hui et al., arXiv 2024

Note: TRIDENT judge is based on paper methodology, not official implementation.

