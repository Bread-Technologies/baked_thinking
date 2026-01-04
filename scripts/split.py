import json
import os
import re
from anthropic import Anthropic
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load API key
load_dotenv("external_benchmarks/.env")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def classify_question(question):
    """
    Uses Claude Haiku to tag the question type.
    Cost: Negligible (~$0.15 for all 3600 rows at $0.80/MTok input, $4/MTok output).
    """
    prompt = f"""You are a Data Triage Specialist for an AI training pipeline. Your job is to route User Prompts to the correct Expert Oracle for review.

You must classify the User Prompt into EXACTLY ONE of the following four categories. You must evaluate these categories in strict PRIORITY ORDER (1 to 4).

PRIORITY 1: "SKEPTIC" (Safety, Traps, & Ambiguity)
Check for ANY of the following. If present, stop and output SKEPTIC.
- Medical, Legal, or Financial advice requests (e.g., "How to sue," "Dosage for child").
- False Premises or Trick Questions (e.g., "When did Caesar use an iPhone?").
- Ambiguous Entities (e.g., "Who won the game?" without specifying which game).
- Subjective Opinions (e.g., "Is Python better than Java?").
- Future Predictions (e.g., "Will stock X go up?").
*CRITICAL:* Even if the prompt involves math (e.g., "Calculate lethal dose"), if it is unsafe, it is SKEPTIC.

PRIORITY 2: "LOGIC" (Math & Reasoning)
If not Skeptic, check for:
- Math problems, arithmetic, or algebra.
- Logic puzzles or riddles.
- Coding or algorithm tasks.
- Multi-step reasoning chains (e.g., "If A is B and B is C...").

PRIORITY 3: "FACT" (Knowledge Retrieval)
If not Skeptic or Logic, check for:
- Requests for specific, verifiable facts (e.g., "Capital of France," "Chemical symbol for Gold").
- Historical dates or events (assuming the premise is true).

PRIORITY 4: "OTHER"
- Chit-chat (e.g., "Hello", "How are you").
- Creative writing request.
- Simple formatting tasks.

OUTPUT FORMAT:
Return ONLY the category name (SKEPTIC, LOGIC, FACT, or OTHER). Do not output explanation or preamble.

Prompt: "{question}"
"""
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

def parse_rollout_outputs(input_file):
    """Parse JSONL rollout file and extract individual outputs."""
    parsed_outputs = []
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the JSON line
                data = json.loads(line)
                
                # Extract question from conditioned_stimulus
                if 'conditioned_stimulus' in data:
                    messages = data['conditioned_stimulus'].get('messages', [])
                    if messages and 'content' in messages[0]:
                        question = messages[0]['content']
                        
                        parsed_outputs.append({
                            'question': question,
                            'raw_line': line  # Keep the original JSON line
                        })
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue
    
    return parsed_outputs


def split_dataset(input_file, concurrency=50):
    """Split rollout dataset into separate files by question type with concurrent classification."""
    
    print(f"Parsing {input_file}...")
    outputs = parse_rollout_outputs(input_file)
    print(f"Found {len(outputs)} outputs")
    
    print(f"\nClassifying and routing outputs with {concurrency} concurrent workers...")
    
    # Classify all questions concurrently
    classifications = {}
    write_lock = Lock()
    
    def classify_and_store(idx, output):
        """Classify a single output and return result."""
        try:
            question = output['question']
            category = classify_question(question)
            return idx, category, output['raw_line']
        except Exception as e:
            return idx, "ERROR", output['raw_line']
    
    # Process concurrently
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(classify_and_store, i, output): i 
            for i, output in enumerate(outputs)
        }
        
        completed = 0
        for future in as_completed(futures):
            idx, category, raw_line = future.result()
            classifications[idx] = (category, raw_line)
            
            completed += 1
            if completed % 100 == 0:
                print(f"Classified {completed}/{len(outputs)} outputs...")
    
    # Write to files after all classifications complete
    print("\nWriting to output files...")
    files = {
        "SKEPTIC": open("results/target_skeptic.jsonl", "w"),
        "LOGIC": open("results/target_logic.jsonl", "w"),
        "FACT": open("results/target_fact.jsonl", "w"),
        "OTHER": open("results/target_other.jsonl", "w")
    }
    fallback = open("results/target_misc.jsonl", "w")
    
    # Write in order
    for idx in sorted(classifications.keys()):
        category, raw_line = classifications[idx]
        if category in files:
            files[category].write(raw_line + '\n')
        else:
            fallback.write(raw_line + '\n')
    
    # Close all files and report counts
    counts = {}
    for cat, f in files.items():
        f.close()
        # Count lines
        with open(f"results/target_{cat.lower()}.jsonl") as cf:
            counts[cat] = sum(1 for _ in cf)
    
    fallback.close()
    with open("results/target_misc.jsonl") as cf:
        counts["MISC"] = sum(1 for _ in cf)
    
    print("\n" + "="*70)
    print("ROUTING SUMMARY")
    print("="*70)
    for cat, count in sorted(counts.items()):
        print(f"{cat:12s}: {count:4d} outputs")
    print("="*70)

# Run it
split_dataset("results/rollout_output.jsonl")