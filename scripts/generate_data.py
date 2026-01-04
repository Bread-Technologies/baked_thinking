#!/usr/bin/env python3
"""
Data Generation Script for Metacognitive Baking.
1. Generates stimuli (questions) based on eval/stimulus_config.py
2. Runs Teacher v4 to generate responses (rollout)
3. Saves raw data for the Oracle Intervention step.
"""

import json
import sys
import os
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.stimulus_config import STIMULUS_CONFIG
from config import BREAD_API_BASE, BREAD_API_KEY, MODEL

# Configuration
OUTPUT_DIR = Path("data/raw_rollouts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = 5  # Parallel requests

def load_teacher_prompt():
    with open("prompts/teacher_v4_thinking.json") as f:
        return json.load(f)["messages"]

def generate_questions(category, config, client, test_mode=False):
    """
    Uses the model to generate questions for a specific category.
    In a real Bread workflow, this would use the 'stim' operation.
    Here we simulate it with a direct call.
    """
    if test_mode:
        print(f"  [TEST MODE] Using hardcoded examples for {category}")
        return config['examples']

    print(f"Generating stimuli for: {category}...")
    
    system_msg = "You are a dataset generator. Output ONLY a JSON array of strings. No markdown, no conversational text."
    user_msg = f"{config['template']}\n\nExamples:\n" + "\n".join([f"- {ex}" for ex in config['examples']])
    
    try:
        response = client.chat.completions.create(
            model="claude-3-5-sonnet-20240620", # Use a smart model for generation
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        content = response.choices[0].message.content.strip()
        # Handle markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        questions = json.loads(content)
        return questions[:config['numq']]
    except Exception as e:
        print(f"Error generating stimuli for {category}: {e}")
        return []

def generate_response(question, teacher_messages, client):
    """
    Runs the Teacher v4 model on a single question.
    """
    messages = teacher_messages + [{"role": "user", "content": question}]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        return {
            "question": question,
            "response": response.choices[0].message.content,
            "model": MODEL
        }
    except Exception as e:
        return {"question": question, "error": str(e)}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Use hardcoded examples instead of generating new ones")
    args = parser.parse_args()

    client = OpenAI(base_url=BREAD_API_BASE, api_key=BREAD_API_KEY)
    teacher_messages = load_teacher_prompt()
    
    all_data = []
    
    # 1. Generate Stimuli
    for category, config in STIMULUS_CONFIG.items():
        questions = generate_questions(category, config, client, test_mode=args.test)
        print(f"  -> Generated {len(questions)} questions for {category}")
        
        # 2. Run Rollout (Teacher v4)
        print(f"  -> Running Teacher v4 on {category}...")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(generate_response, q, teacher_messages, client): q for q in questions}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(questions)):
                results.append({
                    "category": category,
                    **future.result()
                })
        
        all_data.extend(results)
        
        # Save intermediate result
        category_file = OUTPUT_DIR / f"{category}.jsonl"
        with open(category_file, "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")

    print(f"\nDone! Total examples generated: {len(all_data)}")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

