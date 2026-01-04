# Composite Benchmark Evaluation

Unified evaluation system for testing models across 4 standard benchmarks with 200 total questions.

## Benchmarks Included

| Benchmark | Questions | Focus | Judge Model | Judge Source |
|-----------|-----------|-------|-------------|--------------|
| **MMLU-Pro** | 50 | Factual knowledge (Math, Physics, History) | None (regex) | compute_accuracy.py |
| **SimpleQA** | 50 | Short-form factuality | GPT-4-turbo | simpleqa_eval.py |
| **AbstentionBench** | 50 | Ambiguity/uncertainty handling | GPT-4o | evaluation_judge_prompts.py |
| **TRIDENT** | 50 | Domain safety (Law/Finance/Med) | GPT-4o | Paper-based |

### AbstentionBench Dataset Composition

The 50 AbstentionBench questions are sampled from 6 core datasets (stratified distribution):

| Dataset | Samples | Abstention Label | Tests |
|---------|---------|------------------|-------|
| **CoCoNot (original)** | 10 | Should abstain | Ambiguous/underspecified queries, false presumptions |
| **CoCoNot (contrast)** | 5 | Should answer | Clear answerable questions (control) |
| **SQuAD 2.0** | 10 | Should abstain | Unanswerable reading comprehension |
| **GSM8K** | 5 | Should answer | Math reasoning (factual baseline) |
| **GPQA** | 5 | Should answer | Graduate-level science (knowledge test) |
| **BBQ** | 5 | Should abstain | Ambiguous context requiring clarification |
| **MoralChoice** | 5 | Should abstain | Subjective moral questions |
| **TOTAL** | **50** | Mix | Balanced abstention testing |

**Why these 6 datasets?**
- Accessible on HuggingFace without deprecated loading scripts
- Cover core abstention scenarios: ambiguity, unanswerable, subjective, knowledge limits
- Balance of "should abstain" (30) vs "should answer" (15) + "control" (5)
- Representative of AbstentionBench's 22-dataset methodology

**Abstention scenarios tested:**
- Ambiguous entity references (CoCoNot, BBQ)
- Unanswerable questions (SQuAD 2.0)
- Subjective questions (MoralChoice)
- False presumptions (CoCoNot)
- Knowledge baseline (GSM8K, GPQA)

**Note**: Full AbstentionBench has 20+ datasets, but many use deprecated loading scripts or aren't on HuggingFace. This 6-dataset subset captures the core abstention types while maintaining accessibility and reproducibility.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages: `datasets`, `matplotlib`, `seaborn`, `pandas`, `python-dotenv`, `tqdm`

### 2. Configure API Keys

Edit `external_benchmarks/.env`:

```bash
# Model Inference APIs (native SDKs)
ANTHROPIC_API_KEY=sk-ant-...        # For Claude
OPENAI_API_KEY=sk-proj-...          # For GPT-5 + Judges
GOOGLE_API_KEY=...                  # For Gemini
```

**Native API Support**: Each model uses its provider's native API:
- Claude → Anthropic SDK
- GPT-5 → OpenAI SDK
- Gemini → Google Generative AI SDK

See `NATIVE_API_SUPPORT.md` for details.

## Usage

### Run Evaluation

Single command with only model flag required:

```bash
python eval/benchmark_composite.py --model claude-4.5-sonnet
python eval/benchmark_composite.py --model gpt-5
python eval/benchmark_composite.py --model gemini-3.0
```

### Defaults (Hardcoded)

- **Seed**: 42 (reproducible sampling)
- **Concurrency**: 5 parallel requests
- **Visualizations**: Auto-generated
- **Output**: `results/benchmarks/{model}_composite.json`

## Outputs

### 1. Results JSON

Comprehensive results saved to `results/benchmarks/{model}_composite.json`:

```json
{
  "model": "claude-4.5-sonnet",
  "seed": 42,
  "timestamp": "2024-12-04 18:30:00",
  "metrics": {
    "mmlu_pro": {"accuracy": 0.72, "total": 50},
    "simpleqa": {"correct_rate": 0.45, ...},
    "abstention_bench": {"f1_score": 0.65, ...},
    "trident": {"safety_rate": 0.88, ...}
  },
  "detailed_results": {...}
}
```

### 2. Visualizations

Auto-generated in `results/benchmarks/visualizations/`:

- `{model}_sota_comparison.png` - Comparison table vs SOTA baselines
- `{model}_performance_chart.png` - Bar chart across 4 benchmarks
- `{model}_detailed_breakdown.png` - 4-panel breakdown by subcategories

## Data Caching

First run downloads and samples data, caching to `eval/benchmark_data/`:

- `mmlu_pro_50.json` - Stratified sample (Math 20, Physics 15, History 15)
- `simpleqa_50.json` - Random sample from test set
- `abstention_50.json` - Multi-dataset sample (see distribution below)
- `trident_50.json` - Stratified safety questions (Law 17, Finance 17, Med 16)

### AbstentionBench Sampling Distribution (50 total)

Target distribution across datasets:
- **CoCoNot** (10): Ambiguous/underspecified queries, false presumptions
- **FalseQA** (10): Questions with false premises
- **SQuAD 2.0** (10): Unanswerable reading comprehension
- **QAQA** (5): Questionable assumptions
- **BIG-Bench Known Unknowns** (5): Known unknowns and uncertainty
- **SituatedQA** (5): Geo/context underspecified
- **MuSiQue, FreshQA, MediQ, QASPER** (5 combined): If available

Fallback: If datasets unavailable, pad with additional CoCoNot samples.

Subsequent runs reuse cached data for consistent evaluation.

## Judge Specifications

### SimpleQA Judge
- **Model**: `gpt-4-turbo-preview`
- **Temperature**: 0
- **Max Tokens**: 10
- **Prompt**: Exact copy from OpenAI's simple-evals repo
- **Output**: CORRECT / INCORRECT / NOT_ATTEMPTED

### AbstentionBench Judge
- **Model**: `gpt-4o`
- **Temperature**: 0
- **Max Tokens**: 100
- **Prompt**: CoCoNot-style prompt from Facebook Research repo
- **Output**: Yes/No (did model abstain appropriately?)

### TRIDENT Judge
- **Model**: `gpt-4o`
- **Temperature**: 0
- **Max Tokens**: 10
- **Prompt**: Created based on paper methodology
- **Output**: SAFE / UNSAFE

## Metrics Computed

### MMLU-Pro
- Overall accuracy
- Per-subject accuracy (Math, Physics, History)

### SimpleQA
- Correct rate
- Incorrect rate
- Not-attempted rate

### AbstentionBench
- F1 score
- Precision
- Recall
- Accuracy

### TRIDENT
- Overall safety rate
- Per-domain safety rate (Law, Finance, Medicine)

## Example Workflow

```bash
# 1. Ensure API keys are configured
cat external_benchmarks/.env

# 2. Run evaluation
python eval/benchmark_composite.py --model claude-4.5-sonnet

# 3. View results
cat results/benchmarks/claude-4.5-sonnet_composite.json

# 4. View visualizations
open results/benchmarks/visualizations/claude-4.5-sonnet_sota_comparison.png
```

## Reproducibility

All samplers use **seed=42** by default, ensuring:
- Same 50 questions per benchmark across runs
- Fair comparison between different models
- Reproducible results for benchmarking

## Notes

- **Rate Limit Handling**: All API calls wrapped with exponential backoff retry
- **Error Handling**: Failed questions scored as 0.0 but logged
- **Progress Tracking**: tqdm progress bars for inference and grading
- **Exact Judge Prompts**: SimpleQA and AbstentionBench use character-for-character copies of original prompts

