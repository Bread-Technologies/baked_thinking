#!/usr/bin/env python3
"""
SQuAD Token Compliance Checker

Tests a model on 100 SQuAD questions and checks for <confident>/<uncertain> tokens.

Usage:
    python eval/squad_token_check.py --model "Qwen/Qwen3-32B"
    python eval/squad_token_check.py --model "gpt-5"
    python eval/squad_token_check.py --model "claude-sonnet-4-20250514"
"""

import argparse
import os
import re
import random
from pathlib import Path

import datasets
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment
env_path = Path(__file__).parent.parent / "external_benchmarks" / ".env"
load_dotenv(env_path)

# Settings
SEED = 42
NUM_QUESTIONS = 100


def detect_provider(model_name: str) -> str:
    """Auto-detect provider from model name."""
    model_lower = model_name.lower()
    
    if "claude" in model_lower:
        return "anthropic"
    elif "gpt" in model_lower or "o1" in model_lower:
        return "openai"
    elif "gemini" in model_lower:
        return "google"
    else:
        return "bread"


def create_client(provider: str, model_id: str):
    """Create appropriate API client."""
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai
    
    elif provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    elif provider == "bread":
        return OpenAI(
            base_url=os.getenv("BREAD_API_BASE", "https://bapi.bread.com.ai/v1"),
            api_key=os.getenv("BREAD_API_KEY")
        )


def call_model(client, provider: str, model_id: str, question: str) -> str:
    """Call model with question and return response."""
    messages = [{"role": "user", "content": question}]
    
    if provider == "anthropic":
        response = client.messages.create(
            model=model_id,
            messages=messages,
            temperature=1,
            max_tokens=2048
        )
        return response.content[0].text
    
    elif provider == "google":
        model = client.GenerativeModel(model_id)
        response = model.generate_content(
            question,
            generation_config={"temperature": 1, "max_output_tokens": 2048}
        )
        return response.text if hasattr(response, 'text') else str(response)
    
    elif provider in ["openai", "bread"]:
        params = {
            "model": model_id,
            "messages": messages,
            "temperature": 1
        }
        
        # GPT-5 uses max_completion_tokens
        if "gpt-5" in model_id.lower():
            params["max_completion_tokens"] = 2048
        else:
            params["max_tokens"] = 2048
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content or ""


def extract_token(response: str) -> str:
    """Extract confidence token from response."""
    if "<confident>" in response.lower():
        return "confident"
    elif "<uncertain>" in response.lower():
        return "uncertain"
    else:
        return "missing"


def main():
    parser = argparse.ArgumentParser(description="SQuAD Token Compliance Check")
    parser.add_argument("--model", required=True, help="Model name to test")
    args = parser.parse_args()
    
    model_id = args.model
    provider = detect_provider(model_id)
    
    print("="*70)
    print("SQUAD TOKEN COMPLIANCE CHECK")
    print("="*70)
    print(f"Model: {model_id}")
    print(f"Provider: {provider}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Seed: {SEED}")
    print("="*70)
    
    # Load SQuAD 2.0
    print("\n[1/3] Loading SQuAD 2.0 dataset...")
    dataset = datasets.load_dataset("rajpurkar/squad_v2", split="validation")
    
    # Sample 100 questions (mix of answerable and unanswerable)
    rng = random.Random(SEED)
    all_questions = list(dataset)
    sampled = rng.sample(all_questions, NUM_QUESTIONS)
    
    print(f"Sampled {len(sampled)} questions")
    
    # Initialize client
    print("\n[2/3] Initializing API client...")
    client = create_client(provider, model_id)
    print(f"✓ Connected to {provider} API")
    
    # Run inference
    print(f"\n[3/3] Running inference on {NUM_QUESTIONS} questions...")
    
    token_counts = {
        "confident": 0,
        "uncertain": 0,
        "missing": 0
    }
    
    results = []
    
    for item in tqdm(sampled, desc="Processing"):
        question = item["question"]
        
        try:
            response = call_model(client, provider, model_id, question)
            token = extract_token(response)
            token_counts[token] += 1
            
            results.append({
                "question": question,
                "response": response[:200],  # First 200 chars
                "token": token,
                "has_answer": bool(item.get("answers", {}).get("text", []))
            })
        except Exception as e:
            print(f"\nError on question: {e}")
            token_counts["missing"] += 1
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Model: {model_id}")
    print(f"\nToken Distribution:")
    print(f"  <confident>:  {token_counts['confident']:3d} ({token_counts['confident']/NUM_QUESTIONS:.1%})")
    print(f"  <uncertain>:  {token_counts['uncertain']:3d} ({token_counts['uncertain']/NUM_QUESTIONS:.1%})")
    print(f"  Missing:      {token_counts['missing']:3d} ({token_counts['missing']/NUM_QUESTIONS:.1%})")
    print(f"\nTotal: {NUM_QUESTIONS}")
    
    # Compliance rate (has any token)
    compliance = (token_counts['confident'] + token_counts['uncertain']) / NUM_QUESTIONS
    print(f"\n✓ Token Compliance: {compliance:.1%}")
    
    print("="*70)


if __name__ == "__main__":
    main()


