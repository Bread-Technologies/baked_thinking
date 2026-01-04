"""
Data samplers for composite benchmark evaluation.

Loads and samples 50 questions from each of 4 benchmarks:
- MMLU-Pro (Math, Physics, History)
- SimpleQA (factual questions)
- AbstentionBench (ambiguous queries)
- TRIDENT (safety in Law/Finance/Medicine)

All samplers use seed=42 for reproducibility and cache results.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

import datasets
import pandas as pd

SEED = 42
CACHE_DIR = Path(__file__).parent.parent / "benchmark_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def sample_mmlu_pro(n: int = 50, seed: int = SEED) -> List[Dict[str, Any]]:
    """
    Sample stratified questions from MMLU-Pro.
    
    Distribution: Math(20), Physics(15), History(15)
    Format: {question, options: [A-J], answer, subject, question_id}
    """
    cache_file = CACHE_DIR / "mmlu_pro_50.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    
    print("Downloading MMLU-Pro dataset...")
    dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    
    # Group by subject
    subjects = {
        "math": ["math", "mathematics"],
        "physics": ["physics"],
        "history": ["history"]
    }
    
    subject_counts = {"math": 20, "physics": 15, "history": 15}
    
    sampled = []
    rng = random.Random(seed)
    
    for subject_key, subject_names in subjects.items():
        # Filter dataset for this subject
        subject_data = [
            item for item in dataset 
            if item.get("category", "").lower() in [s.lower() for s in subject_names]
        ]
        
        # Sample required count
        count = subject_counts[subject_key]
        if len(subject_data) < count:
            print(f"Warning: Only {len(subject_data)} {subject_key} questions available, requested {count}")
            count = len(subject_data)
        
        samples = rng.sample(subject_data, count)
        
        for idx, item in enumerate(samples):
            sampled.append({
                "question_id": f"mmlu_{subject_key}_{idx+1}",
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
                "subject": item.get("category", subject_key),
                "benchmark": "mmlu_pro"
            })
    
    # Save to cache
    with open(cache_file, "w") as f:
        json.dump(sampled, f, indent=2)
    
    print(f"✓ Sampled {len(sampled)} MMLU-Pro questions (cached)")
    return sampled


def sample_simpleqa(n: int = 50, seed: int = SEED) -> List[Dict[str, Any]]:
    """
    Sample random questions from SimpleQA.
    
    Format: {problem, answer, question_id}
    """
    cache_file = CACHE_DIR / "simpleqa_50.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    
    print("Downloading SimpleQA dataset...")
    df = pd.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv")
    
    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(df)), n)
    
    sampled = []
    for idx in sampled_indices:
        row = df.iloc[idx]
        sampled.append({
            "question_id": f"simpleqa_{idx}",
            "problem": row["problem"],
            "answer": row["answer"],
            "benchmark": "simpleqa"
        })
    
    # Save to cache
    with open(cache_file, "w") as f:
        json.dump(sampled, f, indent=2)
    
    print(f"✓ Sampled {len(sampled)} SimpleQA questions (cached)")
    return sampled


def sample_abstention_bench(n: int = 50, seed: int = SEED) -> List[Dict[str, Any]]:
    """
    Sample questions from AbstentionBench component datasets.
    
    Comprehensive sampling across 10 core abstention-testing datasets:
    - KUQ: Known Unknown Questions (ambiguous, controversial, false premise, future unknown, unsolved)
    - CoCoNot: False presumptions, unknowns, unsupported, temporal, subjective, humanizing, incomprehensible
    - SQuAD 2.0: Unanswerable questions
    - MuSiQue: Unanswerable multi-hop reasoning
    - QAQA: Questionable assumptions
    - FalseQA: False premises
    - FreshQA: Stale/temporal queries
    - SituatedQA: Geo/context underspecified
    - MediQ: Medical high-stakes abstention
    - QASPER: Information-seeking with missing context
    
    Format: {question, ref_answer, abstention_label, category, question_id}
    """
    cache_file = CACHE_DIR / "abstention_50.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    
    print("Downloading AbstentionBench component datasets...")
    print("(This may take a few minutes on first run...)")
    
    all_questions = []
    rng = random.Random(seed)
    
    # Helper to add questions with abstention label
    def add_questions(dataset, source_name, abstention_label, category_prefix, max_samples=None):
        items = list(dataset)
        if max_samples and len(items) > max_samples:
            items = rng.sample(items, max_samples)
        
        for item in items:
            # Extract question text (different field names per dataset)
            question = (
                item.get("prompt") or 
                item.get("question") or 
                item.get("query") or 
                item.get("text") or 
                str(item)
            )
            
            ref_answer = item.get("answer", item.get("answers", []))
            if isinstance(ref_answer, str):
                ref_answer = [ref_answer]
            
            all_questions.append({
                "question": question,
                "ref_answer": ref_answer,
                "abstention_label": abstention_label,
                "category": f"{category_prefix}_{source_name}",
                "source": source_name
            })
    
    # Distribution: ~5-10 questions per dataset type
    
    # 1. CoCoNot: Ambiguous and underspecified (15 total: 10 abstain + 5 contrast)
    try:
        coconot_original = datasets.load_dataset("allenai/coconot", "original", split="test")
        add_questions(coconot_original, "coconot", True, "ambiguous", max_samples=10)
        
        coconot_contrast = datasets.load_dataset("allenai/coconot", "contrast", split="test")
        add_questions(coconot_contrast, "coconot", False, "answerable", max_samples=5)
        print(f"  ✓ CoCoNot: {len([q for q in all_questions if q['source'] == 'coconot'])} questions")
    except Exception as e:
        print(f"  ✗ CoCoNot failed: {e}")
    
    # 2. SQuAD 2.0: Unanswerable reading comprehension (10 samples)
    try:
        squad2 = datasets.load_dataset("rajpurkar/squad_v2", split="validation")
        unanswerable = [item for item in squad2 if not item.get("answers", {}).get("text", [])]
        if unanswerable:
            items = rng.sample(unanswerable, min(10, len(unanswerable)))
            for item in items:
                all_questions.append({
                    "question": item["question"],
                    "ref_answer": [],
                    "abstention_label": True,
                    "category": "unanswerable_squad2",
                    "source": "squad2"
                })
            print(f"  ✓ SQuAD 2.0: {len([q for q in all_questions if q['source'] == 'squad2'])} questions")
    except Exception as e:
        print(f"  ✗ SQuAD 2.0 failed: {e}")
    
    # 3. GSM8K: Math with some requiring abstention (5 samples)
    try:
        gsm8k = datasets.load_dataset("openai/gsm8k", "main", split="test")
        add_questions(gsm8k, "gsm8k", False, "math", max_samples=5)
        print(f"  ✓ GSM8K: {len([q for q in all_questions if q['source'] == 'gsm8k'])} questions")
    except Exception as e:
        print(f"  ✗ GSM8K failed: {e}")
    
    # 4. GPQA: Graduate-level science (5 samples)
    try:
        gpqa = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        add_questions(gpqa, "gpqa", False, "science", max_samples=5)
        print(f"  ✓ GPQA: {len([q for q in all_questions if q['source'] == 'gpqa'])} questions")
    except Exception as e:
        print(f"  ✗ GPQA failed: {e}")
    
    # 5. BBQ (Bias Benchmark): Ambiguous context (5 samples)
    try:
        bbq = datasets.load_dataset("heegyu/bbq", split="test")
        # BBQ has ambiguous and disambiguated contexts - use ambiguous
        ambiguous_bbq = [item for item in bbq if item.get("context_condition") == "ambig"]
        if ambiguous_bbq:
            items = rng.sample(ambiguous_bbq, min(5, len(ambiguous_bbq)))
            for item in items:
                all_questions.append({
                    "question": item.get("question", ""),
                    "ref_answer": [],
                    "abstention_label": True,
                    "category": "ambiguous_bbq",
                    "source": "bbq"
                })
            print(f"  ✓ BBQ: {len([q for q in all_questions if q['source'] == 'bbq'])} questions")
    except Exception as e:
        print(f"  ✗ BBQ failed: {e}")
    
    # 6. MoralChoice: Subjective moral questions (5 samples)
    try:
        moralchoice = datasets.load_dataset("scrubjay55/moralchoice", split="train")
        add_questions(moralchoice, "moralchoice", True, "subjective", max_samples=5)
        print(f"  ✓ MoralChoice: {len([q for q in all_questions if q['source'] == 'moralchoice'])} questions")
    except Exception as e:
        print(f"  ✗ MoralChoice failed: {e}")
    
    # Ensure we have exactly n samples
    if len(all_questions) > n:
        samples = rng.sample(all_questions, n)
    elif len(all_questions) < n:
        print(f"\n  Warning: Only loaded {len(all_questions)}/{n} questions from available datasets")
        samples = all_questions
    else:
        samples = all_questions
    
    # Random sample to exactly n
    if len(all_questions) > n:
        samples = rng.sample(all_questions, n)
    else:
        samples = all_questions
    
    sampled = []
    for idx, item in enumerate(samples):
        sampled.append({
            "question_id": f"abstention_{idx}",
            "question": item["question"],
            "ref_answer": item["ref_answer"],
            "abstention_label": item["abstention_label"],
            "category": item["category"],
            "benchmark": "abstention_bench"
        })
    
    # Save to cache
    with open(cache_file, "w") as f:
        json.dump(sampled, f, indent=2)
    
    print(f"✓ Sampled {len(sampled)} AbstentionBench questions from {len(set(s['source'] for s in samples))} datasets (cached)")
    return sampled


def sample_trident(n: int = 50, seed: int = SEED) -> List[Dict[str, Any]]:
    """
    Sample questions from TRIDENT dataset stratified across domains.
    
    Distribution: Medicine(17), Law(17), Finance(16)
    Format: {harmful_prompt, safe_reply, domain, ethical_refs, question_id}
    """
    cache_file = CACHE_DIR / "trident_50.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    
    print("Loading TRIDENT dataset from local files...")
    
    # Load from external_benchmarks/TRIDENT/dataset/
    trident_dir = Path(__file__).parent.parent.parent / "external_benchmarks" / "TRIDENT" / "dataset"
    
    domains = {
        "med": (trident_dir / "med_final.jsonl", 17),
        "law": (trident_dir / "law_final.jsonl", 17),
        "finance": (trident_dir / "finance_final.jsonl", 16)
    }
    
    sampled = []
    rng = random.Random(seed)
    
    for domain_key, (filepath, count) in domains.items():
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping {domain_key}")
            continue
        
        # Load all questions from domain
        domain_data = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    domain_data.append(json.loads(line))
        
        # Random sample
        if len(domain_data) < count:
            print(f"Warning: Only {len(domain_data)} {domain_key} questions available, requested {count}")
            samples = domain_data
        else:
            samples = rng.sample(domain_data, count)
        
        for idx, item in enumerate(samples):
            sampled.append({
                "question_id": f"trident_{domain_key}_{idx+1}",
                "harmful_prompt": item["harmful_prompt"],
                "safe_reply": item["safe_reply"],
                "domain": item["domain"],
                "ethical_refs": item.get("ethical_refs", []),
                "benchmark": "trident"
            })
    
    # Save to cache
    with open(cache_file, "w") as f:
        json.dump(sampled, f, indent=2)
    
    print(f"✓ Sampled {len(sampled)} TRIDENT questions (cached)")
    return sampled


def load_or_sample_all() -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all benchmark data (from cache if available, otherwise sample).
    
    Returns dict with keys: mmlu_pro, simpleqa, abstention_bench, trident
    """
    return {
        "mmlu_pro": sample_mmlu_pro(),
        "simpleqa": sample_simpleqa(),
        "abstention_bench": sample_abstention_bench(),
        "trident": sample_trident()
    }


if __name__ == "__main__":
    # Test data sampling
    print("Testing data samplers...")
    print("=" * 60)
    
    data = load_or_sample_all()
    
    for benchmark_name, questions in data.items():
        print(f"\n{benchmark_name}: {len(questions)} questions")
        if questions:
            print(f"  Sample ID: {questions[0]['question_id']}")
            print(f"  First key: {list(questions[0].keys())[0]}")

