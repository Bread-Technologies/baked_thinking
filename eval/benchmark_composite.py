#!/usr/bin/env python3
"""
Composite Benchmark Evaluation

Evaluates models on 4 benchmarks (200 questions total):
- MMLU-Pro (50): Factual knowledge (Math, Physics, History)
- SimpleQA (50): Short-form factuality
- AbstentionBench (50): Ambiguity handling
- TRIDENT (50): Domain safety (Law, Finance, Medicine)

Usage:
    python eval/benchmark_composite.py --model claude-4.5-sonnet
    python eval/benchmark_composite.py --model gpt-5
    python eval/benchmark_composite.py --model gemini-3.0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# Add benchmarks module to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.data_samplers import load_or_sample_all
from benchmarks.judges import (
    grade_mmlu_pro,
    grade_simpleqa,
    grade_abstention,
    grade_trident_safety,
    retry_on_rate_limit
)
from benchmarks.visualizer import generate_visualizations

# Load environment variables
env_path = Path(__file__).parent.parent / "external_benchmarks" / ".env"
load_dotenv(env_path)

# Hardcoded defaults
SEED = 42
CONCURRENCY = 10

def detect_provider(model_name: str) -> str:
    """Auto-detect provider from model name."""
    model_lower = model_name.lower()
    
    if "claude" in model_lower:
        return "anthropic"
    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "openai"
    elif "gemini" in model_lower:
        return "google"
    else:
        # Default to Bread API for everything else (Qwen, Llama, etc.)
        return "bread"


def call_anthropic_api(client, model_id: str, messages: List[Dict], temperature: float, max_tokens: int) -> str:
    """Wrapper for Anthropic API."""
    # Convert messages format
    system_msg = None
    user_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        else:
            user_messages.append({"role": msg["role"], "content": msg["content"]})
    
    kwargs = {
        "model": model_id,
        "messages": user_messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if system_msg:
        kwargs["system"] = system_msg
    
    response = client.messages.create(**kwargs)
    return response.content[0].text


def call_google_api(client, model_id: str, messages: List[Dict], temperature: float, max_tokens: int) -> str:
    """Wrapper for Google Gemini API."""
    # Google uses a different format
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += msg["content"]
        elif msg["role"] == "system":
            prompt = msg["content"] + "\n\n" + prompt
    
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens
    }
    
    model = client.GenerativeModel(model_id)
    response = model.generate_content(prompt, generation_config=generation_config)
    
    # Extract text from response
    if hasattr(response, 'text'):
        return response.text
    elif hasattr(response, 'candidates') and response.candidates:
        parts = response.candidates[0].content.parts
        return parts[0].text if parts else ""
    else:
        return str(response)


def call_openai_api(client, model_id: str, messages: List[Dict], temperature: float, max_tokens: int, use_completion_tokens: bool) -> str:
    """Wrapper for OpenAI API."""
    params = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature
    }
    
    if use_completion_tokens:
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens
    
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content or ""


@dataclass
class BenchmarkResult:
    """Result for a single question."""
    question_id: str
    benchmark: str
    question: str
    model_response: str
    grade_result: Dict[str, Any]
    inference_time: float


def run_model_inference(
    question_data: Dict[str, Any],
    model_client: Any,
    provider: str,
    model_id: str
) -> tuple:
    """Run inference on a single question with appropriate API."""
    benchmark = question_data["benchmark"]
    
    # Get question text based on benchmark format
    if benchmark == "mmlu_pro":
        # Format as multiple choice with options A-J
        question_text = question_data["question"] + "\n\nOptions:\n"
        for i, option in enumerate(question_data["options"]):
            letter = chr(65 + i)  # A=65, B=66, etc.
            question_text += f"{letter}. {option}\n"
        question_text += "\nAnswer with the letter (A-J) only."
    elif benchmark == "simpleqa":
        question_text = question_data["problem"]
    elif benchmark == "abstention_bench":
        question_text = question_data["question"]
    elif benchmark == "trident":
        question_text = question_data["harmful_prompt"]
    else:
        question_text = str(question_data)
    
    messages = [{"role": "user", "content": question_text}]
    start_time = time.time()
    
    # Route to appropriate API
    try:
        if provider == "anthropic":
            model_response = retry_on_rate_limit(
                lambda: call_anthropic_api(model_client, model_id, messages, temperature=1, max_tokens=10000)
            )
        elif provider == "google":
            model_response = retry_on_rate_limit(
                lambda: call_google_api(model_client, model_id, messages, temperature=1, max_tokens=10000)
            )
        elif provider == "openai":
            use_completion_tokens = "gpt-5" in model_id.lower()
            model_response = retry_on_rate_limit(
                lambda: call_openai_api(model_client, model_id, messages, temperature=1, max_tokens=10000, use_completion_tokens=use_completion_tokens)
            )
        elif provider == "bread":
            # Use OpenAI client with Bread API (no completion_tokens)
            model_response = retry_on_rate_limit(
                lambda: call_openai_api(model_client, model_id, messages, temperature=1, max_tokens=10000, use_completion_tokens=False)
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        model_response = f"ERROR: {e}"
    
    inference_time = time.time() - start_time
    
    return question_data["question_id"], model_response, inference_time


def run_inference_batch(
    all_questions: List[Dict[str, Any]],
    model_client: Any,
    provider: str,
    model_id: str,
    concurrency: int = CONCURRENCY
) -> Dict[str, tuple]:
    """
    Run inference on all questions with concurrency.
    Returns: {question_id: (response, inference_time)}
    """
    results = {}
    print_lock = Lock()
    
    print(f"\nRunning inference on {len(all_questions)} questions...")
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(run_model_inference, q, model_client, provider, model_id): q
            for q in all_questions
        }
        
        with tqdm(total=len(all_questions), desc="Inference") as pbar:
            for future in as_completed(futures):
                try:
                    q_id, response, inf_time = future.result()
                    results[q_id] = (response, inf_time)
                    pbar.update(1)
                except Exception as e:
                    q = futures[future]
                    print(f"\nError on {q['question_id']}: {e}")
                    results[q['question_id']] = (f"ERROR: {e}", 0.0)
                    pbar.update(1)
    
    return results


def evaluate_benchmark(
    benchmark_name: str,
    questions: List[Dict[str, Any]],
    responses: Dict[str, tuple],
    judge_client: OpenAI
) -> List[BenchmarkResult]:
    """Grade responses using appropriate judge for benchmark."""
    results = []
    
    print(f"\nGrading {benchmark_name} ({len(questions)} questions)...")
    
    for q in tqdm(questions, desc=f"Grading {benchmark_name}"):
        q_id = q["question_id"]
        model_response, inf_time = responses.get(q_id, ("", 0.0))
        
        try:
            # Route to correct grader
            if benchmark_name == "mmlu_pro":
                grade_result = grade_mmlu_pro(
                    q["question"],
                    q["options"],
                    q["answer"],
                    model_response
                )
                question_text = q["question"]
                
            elif benchmark_name == "simpleqa":
                grade_result = grade_simpleqa(
                    q["problem"],
                    q["answer"],
                    model_response,
                    judge_client
                )
                question_text = q["problem"]
                
            elif benchmark_name == "abstention_bench":
                grade_result = grade_abstention(
                    q["question"],
                    q["ref_answer"],
                    q["abstention_label"],
                    model_response,
                    judge_client
                )
                question_text = q["question"]
                
            elif benchmark_name == "trident":
                grade_result = grade_trident_safety(
                    q["harmful_prompt"],
                    q["safe_reply"],
                    q["domain"],
                    q["ethical_refs"],
                    model_response,
                    judge_client
                )
                question_text = q["harmful_prompt"]
            else:
                raise ValueError(f"Unknown benchmark: {benchmark_name}")
            
            results.append(BenchmarkResult(
                question_id=q_id,
                benchmark=benchmark_name,
                question=question_text[:200],  # Truncate for storage
                model_response=model_response[:500],  # Truncate for storage
                grade_result=grade_result,
                inference_time=inf_time
            ))
            
        except Exception as e:
            print(f"\nError grading {q_id}: {e}")
            results.append(BenchmarkResult(
                question_id=q_id,
                benchmark=benchmark_name,
                question=str(q)[:200],
                model_response=model_response[:500],
                grade_result={"error": str(e), "score": 0.0},
                inference_time=inf_time
            ))
    
    return results


def compute_metrics(results_by_benchmark: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
    """Compute aggregate metrics per benchmark."""
    metrics = {}
    
    # MMLU-Pro: Accuracy
    mmlu_results = results_by_benchmark.get("mmlu_pro", [])
    if mmlu_results:
        correct = sum(1 for r in mmlu_results if r.grade_result.get("correct", False))
        metrics["mmlu_pro"] = {
            "accuracy": correct / len(mmlu_results),
            "total": len(mmlu_results),
            "correct": correct
        }
    
    # SimpleQA: Correct/Incorrect/Not-Attempted rates
    simpleqa_results = results_by_benchmark.get("simpleqa", [])
    if simpleqa_results:
        correct = sum(1 for r in simpleqa_results if r.grade_result.get("correct", False))
        incorrect = sum(1 for r in simpleqa_results if r.grade_result.get("incorrect", False))
        not_attempted = sum(1 for r in simpleqa_results if r.grade_result.get("not_attempted", False))
        
        metrics["simpleqa"] = {
            "correct_rate": correct / len(simpleqa_results),
            "incorrect_rate": incorrect / len(simpleqa_results),
            "not_attempted_rate": not_attempted / len(simpleqa_results),
            "total": len(simpleqa_results),
            "correct": correct,
            "incorrect": incorrect,
            "not_attempted": not_attempted
        }
    
    # AbstentionBench: Precision, Recall, F1
    abstention_results = results_by_benchmark.get("abstention_bench", [])
    if abstention_results:
        tp = sum(1 for r in abstention_results if r.grade_result.get("true_positive", False))
        tn = sum(1 for r in abstention_results if r.grade_result.get("true_negative", False))
        fp = sum(1 for r in abstention_results if r.grade_result.get("false_positive", False))
        fn = sum(1 for r in abstention_results if r.grade_result.get("false_negative", False))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(abstention_results) if abstention_results else 0.0
        
        metrics["abstention_bench"] = {
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "total": len(abstention_results),
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn
        }
    
    # TRIDENT: Safety rate (% of safe responses)
    trident_results = results_by_benchmark.get("trident", [])
    if trident_results:
        safe_count = sum(1 for r in trident_results if r.grade_result.get("safe", False))
        
        # By domain breakdown
        by_domain = {}
        for r in trident_results:
            domain = "unknown"
            for q in results_by_benchmark.get("trident", []):
                if q.question_id == r.question_id:
                    # Extract domain from question_id (e.g., "trident_med_1")
                    parts = r.question_id.split("_")
                    if len(parts) >= 2:
                        domain = parts[1]
                    break
            
            if domain not in by_domain:
                by_domain[domain] = {"safe": 0, "total": 0}
            by_domain[domain]["total"] += 1
            if r.grade_result.get("safe", False):
                by_domain[domain]["safe"] += 1
        
        for domain in by_domain:
            by_domain[domain]["rate"] = by_domain[domain]["safe"] / by_domain[domain]["total"]
        
        metrics["trident"] = {
            "safety_rate": safe_count / len(trident_results),
            "total": len(trident_results),
            "safe": safe_count,
            "unsafe": len(trident_results) - safe_count,
            "by_domain": by_domain
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Composite Benchmark Evaluation")
    parser.add_argument(
        "--model", 
        required=True,
        help="Model name to evaluate (e.g., 'gpt-5', 'claude-sonnet-4-20250514', 'Qwen/Qwen3-32B')"
    )
    args = parser.parse_args()
    
    # Auto-detect provider from model name
    model_id = args.model
    provider = detect_provider(model_id)
    
    print("=" * 70)
    print("COMPOSITE BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Provider: {provider}")
    print(f"Seed: {SEED} (reproducible sampling)")
    print(f"Concurrency: {CONCURRENCY}")
    print("=" * 70)
    
    # 1. Load benchmark data (cached with seed=42)
    print("\n[1/6] Loading benchmark data...")
    benchmark_data = load_or_sample_all()
    
    total_questions = sum(len(q) for q in benchmark_data.values())
    print(f"Loaded {total_questions} total questions:")
    for name, questions in benchmark_data.items():
        print(f"  - {name}: {len(questions)} questions")
    
    # 2. Initialize API clients
    print("\n[2/6] Initializing API clients...")
    
    # Initialize model client based on provider
    if provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        model_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        print(f"  Model API: Anthropic ({model_id})")
        
    elif provider == "google":
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model_client = genai  # Pass the module itself
        print(f"  Model API: Google ({model_id})")
        
    elif provider == "openai":
        model_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"  Model API: OpenAI ({model_id})")
        
    elif provider == "bread":
        model_client = OpenAI(
            base_url=os.getenv("BREAD_API_BASE", "https://bapi.bread.com.ai/v1"),
            api_key=os.getenv("BREAD_API_KEY")
        )
        print(f"  Model API: Bread ({model_id})")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Initialize judge client (always OpenAI)
    judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"  Judge API: OpenAI (GPT-4-turbo, GPT-4o)")
    
    # 3. Run inference on all questions
    print("\n[3/6] Running model inference...")
    all_questions = []
    for questions in benchmark_data.values():
        all_questions.extend(questions)
    
    responses = run_inference_batch(all_questions, model_client, provider, model_id, CONCURRENCY)
    
    # 4. Grade with appropriate judges
    print("\n[4/6] Grading responses with benchmark-specific judges...")
    all_results = {}
    
    for benchmark_name, questions in benchmark_data.items():
        all_results[benchmark_name] = evaluate_benchmark(
            benchmark_name,
            questions,
            responses,
            judge_client
        )
    
    # 5. Compute metrics
    print("\n[5/6] Computing metrics...")
    metrics = compute_metrics(all_results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    if "mmlu_pro" in metrics:
        print(f"\nMMLU-Pro:")
        print(f"  Accuracy: {metrics['mmlu_pro']['accuracy']:.1%} ({metrics['mmlu_pro']['correct']}/{metrics['mmlu_pro']['total']})")
    
    if "simpleqa" in metrics:
        print(f"\nSimpleQA:")
        print(f"  Correct: {metrics['simpleqa']['correct_rate']:.1%}")
        print(f"  Incorrect: {metrics['simpleqa']['incorrect_rate']:.1%}")
        print(f"  Not Attempted: {metrics['simpleqa']['not_attempted_rate']:.1%}")
    
    if "abstention_bench" in metrics:
        print(f"\nAbstentionBench:")
        print(f"  F1 Score: {metrics['abstention_bench']['f1_score']:.3f}")
        print(f"  Precision: {metrics['abstention_bench']['precision']:.3f}")
        print(f"  Recall: {metrics['abstention_bench']['recall']:.3f}")
        print(f"  Accuracy: {metrics['abstention_bench']['accuracy']:.1%}")
    
    if "trident" in metrics:
        print(f"\nTRIDENT:")
        print(f"  Safety Rate: {metrics['trident']['safety_rate']:.1%} ({metrics['trident']['safe']}/{metrics['trident']['total']})")
        if "by_domain" in metrics["trident"]:
            for domain, stats in metrics["trident"]["by_domain"].items():
                print(f"    {domain}: {stats['rate']:.1%} ({stats['safe']}/{stats['total']})")
    
    print("=" * 70)
    
    # 6. Save results
    print("\n[6/6] Saving results and generating visualizations...")
    output_dir = Path("results/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize model name for filename (replace slashes and special chars)
    safe_model_name = model_id.replace("/", "_").replace(" ", "_")
    output_file = output_dir / f"{safe_model_name}_composite.json"
    
    output_data = {
        "model": model_id,
        "provider": provider,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total_questions,
        "metrics": metrics,
        "detailed_results": {
            name: [asdict(r) for r in results]
            for name, results in all_results.items()
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved: {output_file}")
    
    # Generate visualizations
    viz_files = generate_visualizations(safe_model_name, metrics, all_results)
    print(f"✓ Visualizations generated:")
    for viz_file in viz_files:
        print(f"  - {viz_file}")
    
    print("\n✅ Evaluation complete!")
    return output_data


if __name__ == "__main__":
    main()

