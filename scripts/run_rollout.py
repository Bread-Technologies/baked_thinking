#!/usr/bin/env python3
"""
Rollout Script for Metacognitive Baking.
1. Reads stimuli from Bread output JSON.
2. Runs Teacher v4 to generate responses.
3. Saves raw data for the Oracle Intervention step.
"""

import json
import sys
import re
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BREAD_API_BASE, BREAD_API_KEY, MODEL

# Configuration
INPUT_FILE = Path("results/stim_output_20251203_171036.json")
OUTPUT_DIR = Path("data/raw_rollouts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "rollout_v4.jsonl"
MAX_WORKERS = 10  # Parallel requests

def load_teacher_prompt():
    with open("prompts/teacher_v4_thinking.json") as f:
        return json.load(f)["messages"]

def load_stimuli(path: Path):
    """Parse the messy stim output file."""
    print(f"Loading stimuli from {path}...")
    with open(path) as f:
        content = f.read()
    
    # Find the start of the JSON object
    match = re.search(r'\{', content)
    if not match:
        raise ValueError("No JSON object found in file")
    
    json_content = content[match.start():]
    data = json.loads(json_content)
    
    stimuli = []
    for item in data.get("output", []):
        # Extract the question from the messages
        # Structure: unconditioned_stimulus -> messages -> list of {role, content}
        try:
            msgs = item.get("unconditioned_stimulus", {}).get("messages", [])
            # The last user message is usually the question
            user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
            if user_msgs:
                stimuli.append(user_msgs[-1])
        except Exception as e:
            print(f"Skipping malformed item: {e}")
            continue
            
    print(f"Found {len(stimuli)} valid stimuli.")
    return stimuli

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
    client = OpenAI(base_url=BREAD_API_BASE, api_key=BREAD_API_KEY)
    teacher_messages = load_teacher_prompt()
    
    stimuli = load_stimuli(INPUT_FILE)
    
    print(f"Running Teacher v4 rollout on {len(stimuli)} questions...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_response, q, teacher_messages, client): q for q in stimuli}
        
        # Use tqdm for progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(stimuli)):
            results.append(future.result())
    
    # Save results
    print(f"Saving {len(results)} responses to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print("Done!")

if __name__ == "__main__":
    main()


