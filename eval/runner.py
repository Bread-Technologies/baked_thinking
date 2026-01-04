#!/usr/bin/env python3
"""
Batch evaluation runner for metacognitive testing.

Runs all test questions through a model and computes metrics.

Usage:
    python eval/runner.py                           # Test prompted model
    python eval/runner.py --base                    # Test base model (no prompt)
    python eval/runner.py --output results.json    # Save results to file
    python eval/runner.py --categories factual_easy ambiguous_entity  # Test specific categories
"""

import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable, TypeVar

from openai import OpenAI, RateLimitError, APIError

# Add parent/scripts to path for config
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from config import BREAD_API_BASE, BREAD_API_KEY, MODEL


T = TypeVar('T')


def retry_on_rate_limit(
    func: Callable[[], T],
    max_retries: int = 10,
    initial_wait: float = 5.0,
    max_wait: float = 120.0,
    backoff_factor: float = 1.5,
) -> T:
    """
    Retry a function call with exponential backoff on rate limit errors (429).
    
    Args:
        func: Function to call (should take no args - use lambda if needed)
        max_retries: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time between retries
        backoff_factor: Multiplier for wait time after each retry
    """
    wait_time = initial_wait
    
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt
            
            print(f"\n⚠️  Rate limit hit (429). Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait_time)
            wait_time = min(wait_time * backoff_factor, max_wait)
        except APIError as e:
            # Also handle other API errors with retry
            if "429" in str(e) or "rate" in str(e).lower():
                if attempt == max_retries - 1:
                    raise
                print(f"\n⚠️  API error (likely rate limit). Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                wait_time = min(wait_time * backoff_factor, max_wait)
            else:
                raise  # Re-raise non-rate-limit errors immediately
    
    raise RuntimeError("Retry logic failed unexpectedly")


@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    category: str
    question_text: str
    expected_token: str
    response: str
    detected_token: Optional[str]
    token_correct: bool
    has_token: bool


@dataclass 
class CategoryMetrics:
    """Metrics for a single category."""
    category: str
    total: int
    has_token: int
    token_compliance: float
    correct_token: int
    accuracy: float


@dataclass
class EvalResults:
    """Full evaluation results."""
    model: str
    mode: str  # "prompted" or "base"
    prompt_version: str
    timestamp: str
    total_questions: int
    overall_token_compliance: float
    overall_accuracy: float
    category_metrics: dict
    question_results: list


def load_questions(path: str) -> dict:
    """Load test questions from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_prompt(path: str) -> list[dict]:
    """Load teacher prompt messages."""
    with open(path) as f:
        data = json.load(f)
    return data["messages"]


def parse_confidence_token(response: str) -> Optional[str]:
    """
    Extract confidence token from response.
    Returns 'confident', 'uncertain', or None if not found.
    
    Important: Strips out <think>...</think> blocks first since Qwen
    includes reasoning that might mention confidence tokens.
    """
    # Strip out <think>...</think> blocks - they contain reasoning, not the final answer
    clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    response_lower = clean_response.lower()
    
    # Look for explicit tokens (check uncertain first since it's more specific)
    if "<uncertain>" in response_lower:
        return "uncertain"
    if "<confident>" in response_lower:
        return "confident"
    
    # Fallback: look for variations
    patterns = [
        (r"<\s*uncertain\s*>", "uncertain"),
        (r"<\s*confident\s*>", "confident"),
        (r"\[uncertain\]", "uncertain"),
        (r"\[confident\]", "confident"),
    ]
    
    for pattern, token in patterns:
        if re.search(pattern, response_lower):
            return token
    
    return None


def run_question(
    client: OpenAI,
    question: str,
    teacher_messages: list[dict],
    base_mode: bool = False,
) -> str:
    """Run a single question through the model, with retry on rate limits."""
    
    if base_mode:
        messages = [{"role": "user", "content": question}]
    else:
        messages = teacher_messages + [{"role": "user", "content": question}]
    
    # Wrap API call with rate limit retry logic
    response = retry_on_rate_limit(
        lambda: client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
    )
    
    return response.choices[0].message.content


def evaluate_questions(
    client: OpenAI,
    questions_data: dict,
    teacher_messages: list[dict],
    base_mode: bool = False,
    categories: Optional[list[str]] = None,
    verbose: bool = True,
) -> EvalResults:
    """Run evaluation on all questions."""
    
    results = []
    category_results = {}
    
    categories_to_test = categories or list(questions_data["categories"].keys())
    
    for cat_name in categories_to_test:
        if cat_name not in questions_data["categories"]:
            print(f"Warning: Category '{cat_name}' not found, skipping")
            continue
            
        cat_data = questions_data["categories"][cat_name]
        expected = cat_data["expected_token"]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Category: {cat_name}")
            print(f"Expected token: {expected}")
            print(f"{'='*60}")
        
        cat_results = []
        
        for q in cat_data["questions"]:
            q_id = q["id"]
            q_text = q["text"]
            
            if verbose:
                print(f"\n[{q_id}] {q_text[:60]}...")
            
            try:
                response = run_question(client, q_text, teacher_messages, base_mode)
                detected = parse_confidence_token(response)
                has_token = detected is not None
                token_correct = detected == expected
                
                if verbose:
                    token_display = detected or "NONE"
                    status = "✓" if token_correct else "✗"
                    print(f"    Token: {token_display} {status}")
                
                result = QuestionResult(
                    question_id=q_id,
                    category=cat_name,
                    question_text=q_text,
                    expected_token=expected,
                    response=response,
                    detected_token=detected,
                    token_correct=token_correct,
                    has_token=has_token,
                )
                results.append(result)
                cat_results.append(result)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        # Compute category metrics
        if cat_results:
            total = len(cat_results)
            has_token_count = sum(1 for r in cat_results if r.has_token)
            correct_count = sum(1 for r in cat_results if r.token_correct)
            
            category_results[cat_name] = CategoryMetrics(
                category=cat_name,
                total=total,
                has_token=has_token_count,
                token_compliance=has_token_count / total,
                correct_token=correct_count,
                accuracy=correct_count / total,
            )
    
    # Compute overall metrics
    total_questions = len(results)
    overall_compliance = sum(1 for r in results if r.has_token) / total_questions if total_questions > 0 else 0
    overall_accuracy = sum(1 for r in results if r.token_correct) / total_questions if total_questions > 0 else 0
    
    return EvalResults(
        model=MODEL,
        mode="base" if base_mode else "prompted",
        prompt_version=questions_data.get("version", "unknown"),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        total_questions=total_questions,
        overall_token_compliance=overall_compliance,
        overall_accuracy=overall_accuracy,
        category_metrics={k: asdict(v) for k, v in category_results.items()},
        question_results=[asdict(r) for r in results],
    )


def print_summary(results: EvalResults):
    """Print evaluation summary."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {results.model}")
    print(f"Mode: {results.mode}")
    print(f"Timestamp: {results.timestamp}")
    print(f"Total Questions: {results.total_questions}")
    print(f"\nOverall Token Compliance: {results.overall_token_compliance:.1%}")
    print(f"Overall Token Accuracy: {results.overall_accuracy:.1%}")
    
    print("\n" + "-" * 70)
    print(f"{'Category':<25} {'Total':>6} {'Compliance':>12} {'Accuracy':>10}")
    print("-" * 70)
    
    for cat_name, metrics in results.category_metrics.items():
        print(f"{cat_name:<25} {metrics['total']:>6} {metrics['token_compliance']:>11.1%} {metrics['accuracy']:>9.1%}")
    
    print("-" * 70)
    
    # Highlight failures
    failures = [r for r in results.question_results if not r["token_correct"]]
    if failures:
        print(f"\nFailed questions ({len(failures)}):")
        for f in failures[:10]:  # Show first 10
            detected = f["detected_token"] or "NONE"
            print(f"  [{f['question_id']}] Expected: {f['expected_token']}, Got: {detected}")
            print(f"      Q: {f['question_text'][:50]}...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch evaluation runner")
    parser.add_argument("--base", action="store_true", help="Test base model (no teacher prompt)")
    parser.add_argument("--prompt", default="prompts/teacher_v1.json", help="Path to teacher prompt")
    parser.add_argument("--questions", default="eval/test_questions.json", help="Path to test questions")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--categories", nargs="+", help="Specific categories to test")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()
    
    # Resolve paths
    base_path = Path(__file__).parent.parent
    prompt_path = base_path / args.prompt
    questions_path = base_path / args.questions
    
    # Initialize
    client = OpenAI(base_url=BREAD_API_BASE, api_key=BREAD_API_KEY)
    
    # Load data
    questions_data = load_questions(questions_path)
    teacher_messages = [] if args.base else load_prompt(prompt_path)
    
    print(f"Testing mode: {'BASE' if args.base else 'PROMPTED'}")
    print(f"Model: {MODEL}")
    if not args.base:
        print(f"Prompt: {args.prompt}")
    
    # Run evaluation
    results = evaluate_questions(
        client=client,
        questions_data=questions_data,
        teacher_messages=teacher_messages,
        base_mode=args.base,
        categories=args.categories,
        verbose=not args.quiet,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        output_path = base_path / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

