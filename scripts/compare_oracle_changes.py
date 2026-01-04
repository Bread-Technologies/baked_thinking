#!/usr/bin/env python3
"""
Compare original vs oracle-cleaned responses.

Shows side-by-side diff of what the oracles changed.

Usage:
    python scripts/compare_oracle_changes.py --num 10
    python scripts/compare_oracle_changes.py --category logic --num 5
"""

import json
import random
import argparse
import re
from pathlib import Path


def load_jsonl(filepath):
    """Load JSONL file into list."""
    outputs = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                outputs.append(json.loads(line))
    return outputs


def extract_question_and_response(data):
    """Extract question and response from trajectory."""
    question = data['conditioned_stimulus']['messages'][0]['content']
    response = data['response']['content']
    return question, response


def find_matching_original(cleaned_data, original_outputs):
    """Find the original output matching a cleaned output."""
    cleaned_q = cleaned_data['conditioned_stimulus']['messages'][0]['content']
    
    for orig in original_outputs:
        orig_q = orig['conditioned_stimulus']['messages'][0]['content']
        if orig_q == cleaned_q:
            return orig
    return None


def display_comparison(idx, question, original_response, cleaned_response):
    """Display full original and cleaned responses."""
    print("\n" + "="*80)
    print(f"COMPARISON {idx}")
    print("="*80)
    print(f"Question: {question}")
    
    # Check if they're different first
    if original_response == cleaned_response:
        print("\n⚠️  NO CHANGE (oracle kept original)\n")
        return
    
    print("\n" + "-"*80)
    print("ORIGINAL (Qwen):")
    print("-"*80)
    print(original_response)
    
    print("\n" + "-"*80)
    print("ORACLE-CLEANED:")
    print("-"*80)
    print(cleaned_response)
    
    print("\n✅ CHANGED")
    
    # Check token changes
    orig_token = "confident" if "<confident>" in original_response else "uncertain" if "<uncertain>" in original_response else "missing"
    clean_token = "confident" if "<confident>" in cleaned_response else "uncertain" if "<uncertain>" in cleaned_response else "missing"
    
    if orig_token != clean_token:
        print(f"🔄 TOKEN CHANGED: {orig_token} → {clean_token}")
    
    # Show length difference
    print(f"📊 Length: {len(original_response)} → {len(cleaned_response)} ({len(cleaned_response) - len(original_response):+d} chars)")


def main():
    parser = argparse.ArgumentParser(description="Compare original vs oracle-cleaned responses")
    parser.add_argument("--num", "-n", type=int, default=10, help="Number of samples to compare")
    parser.add_argument("--category", choices=["logic", "skeptic", "fact", "other", "all"], default="all",
                       help="Which category to sample from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    results_dir = Path("results")
    
    # Load original data
    print("Loading original rollout data...")
    original_outputs = load_jsonl(results_dir / "rollout_output.jsonl")
    print(f"  Loaded {len(original_outputs)} original outputs")
    
    # Determine which categories to compare
    if args.category == "all":
        categories = ["logic", "skeptic", "fact", "other"]
        samples_per_cat = max(2, args.num // 4)
    else:
        categories = [args.category]
        samples_per_cat = args.num
    
    random.seed(args.seed)
    total_compared = 0
    
    for category in categories:
        cleaned_file = results_dir / f"cleaned_{category}.jsonl"
        
        if not cleaned_file.exists():
            print(f"\n⚠️  Skipping {category}: cleaned file not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*80}")
        
        # Load cleaned data
        cleaned_outputs = load_jsonl(cleaned_file)
        print(f"Loaded {len(cleaned_outputs)} cleaned outputs")
        
        # Sample
        if len(cleaned_outputs) <= samples_per_cat:
            samples = cleaned_outputs
        else:
            samples = random.sample(cleaned_outputs, samples_per_cat)
        
        # Compare each sample
        for i, cleaned_data in enumerate(samples, 1):
            question, cleaned_response = extract_question_and_response(cleaned_data)
            
            # Find matching original
            original_data = find_matching_original(cleaned_data, original_outputs)
            
            if original_data:
                _, original_response = extract_question_and_response(original_data)
                display_comparison(total_compared + 1, question, original_response, cleaned_response)
                total_compared += 1
            else:
                print(f"\n⚠️  Could not find matching original for sample {i}")
    
    print("\n" + "="*80)
    print(f"Compared {total_compared} outputs total")
    print("="*80)


if __name__ == "__main__":
    main()

