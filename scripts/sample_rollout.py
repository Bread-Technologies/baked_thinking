#!/usr/bin/env python3
"""
Sample random outputs from rollout data.

Usage:
    python scripts/sample_rollout.py results/rollout_output_20251203_211812.json
    python scripts/sample_rollout.py results/rollout_output_20251203_211812.json --num 10
    python scripts/sample_rollout.py results/rollout_output_20251203_211812.json --save samples.json
"""

import json
import random
import argparse
import re
from pathlib import Path


def parse_rollout_file(filepath):
    """
    Parse rollout output file and extract individual outputs.
    
    Returns list of simplified output dicts with question, response, confidence.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by output markers "--- Output N ---"
    output_chunks = re.split(r'--- Output \d+ ---', content)[1:]
    
    outputs = []
    
    for chunk in output_chunks:
        try:
            # Extract question from conditioned_stimulus
            question_match = re.search(
                r'"conditioned_stimulus":\s*\{[^}]*"messages":\s*\[\s*\{[^}]*"content":\s*"([^"]+)"',
                chunk,
                re.DOTALL
            )
            question = question_match.group(1) if question_match else "N/A"
            
            # Extract response content (look for response.content field)
            response_match = re.search(
                r'"response":\s*\{[^}]*"content":\s*"(.*?)"(?=,\s*"usage")',
                chunk,
                re.DOTALL
            )
            response = response_match.group(1) if response_match else "N/A"
            
            # Clean up escape sequences
            response = response.replace('\\n', '\n').replace('\\"', '"')
            question = question.replace('\\n', '\n').replace('\\"', '"')
            
            # Extract confidence token
            confidence = "not found"
            if '<confident>' in response:
                confidence = "<confident>"
            elif '<uncertain>' in response:
                confidence = "<uncertain>"
            
            # Extract token count
            tokens_match = re.search(r'"total_tokens":\s*(\d+)', chunk)
            total_tokens = int(tokens_match.group(1)) if tokens_match else None
            
            outputs.append({
                'question': question,
                'response': response,
                'confidence': confidence,
                'total_tokens': total_tokens
            })
        except Exception as e:
            # Skip outputs that fail to parse
            continue
    
    return outputs


def sample_outputs(outputs, n=10, seed=42):
    """Randomly sample n outputs."""
    random.seed(seed)
    
    if len(outputs) <= n:
        return outputs
    
    return random.sample(outputs, n)


def display_output(output, index):
    """Display a single output in readable format."""
    print(f"\n{'='*70}")
    print(f"SAMPLE {index}")
    print(f"{'='*70}")
    
    # Display question
    question = output.get('question', 'N/A')
    print(f"Question: {question}")
    
    # Display FULL response
    response = output.get('response', 'N/A')
    print(f"\nResponse (FULL):")
    print(response)
    
    # Display confidence
    confidence = output.get('confidence', 'not found')
    print(f"\nConfidence: {confidence}")
    
    # Display token count
    tokens = output.get('total_tokens')
    if tokens:
        print(f"Tokens: {tokens} total")


def main():
    parser = argparse.ArgumentParser(description="Sample random outputs from rollout data")
    parser.add_argument("input_file", help="Path to rollout output JSON file")
    parser.add_argument("--num", "-n", type=int, default=10, help="Number of samples (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--save", help="Save samples to JSON file")
    args = parser.parse_args()
    
    # Parse file
    print(f"Parsing {args.input_file}...")
    outputs = parse_rollout_file(args.input_file)
    print(f"Found {len(outputs)} outputs")
    
    # Sample
    samples = sample_outputs(outputs, n=args.num, seed=args.seed)
    print(f"Sampled {len(samples)} outputs (seed={args.seed})")
    
    # Display
    for i, output in enumerate(samples, 1):
        display_output(output, i)
    
    # Save if requested
    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"\n✓ Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()

