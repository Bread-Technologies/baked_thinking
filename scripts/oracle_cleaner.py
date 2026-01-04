import json
import os
import re
from typing import Dict, Any
from tqdm import tqdm
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

# Load environment variables
load_dotenv("external_benchmarks/.env")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# API KEYS (Set these in your environment or paste them here)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# MODEL NAMES (Updated to latest versions)
MODEL_LOGIC = "gpt-5"                           # The Logic Oracle
MODEL_SKEPTIC = "claude-sonnet-4-5-20250929"      # The Truth/Safety Auditor (Claude 4.5 Sonnet)
MODEL_FACT = "gemini-3-pro-preview"             # The Fact Engine (Gemini 3)

# CONCURRENCY SETTINGS
CONCURRENCY = 50  # Parallel workers for API calls

# INPUT/OUTPUT FILES (Updated to match split.py output)
FILES_CONFIG = [
    {
        "type": "logic",
        "input": "results/target_logic.jsonl",
        "output": "results/cleaned_logic.jsonl"
    },
    {
        "type": "skeptic",
        "input": "results/target_skeptic.jsonl",
        "output": "results/cleaned_skeptic.jsonl"
    },
    {
        "type": "fact",
        "input": "results/target_fact.jsonl",
        "output": "results/cleaned_fact.jsonl"
    },
    {
        "type": "other",
        "input": "results/target_other.jsonl",
        "output": "results/cleaned_other.jsonl"
    }
]

# ==============================================================================
# SYSTEM PROMPTS (The "Brain")
# ==============================================================================

PROMPT_LOGIC = """You are a Senior Logic Editor reviewing a draft response to a math/logic problem.

INPUT:
1. User Query
2. Student's Draft Response (<think> block + Answer + Token)

INSTRUCTIONS:
1. Solve the problem yourself to establish Ground Truth.
2. Review the Student's logic.
   - If the Student is WRONG: Rewrite the <think> trace showing the model identifying the error (e.g., "Wait, I need to check units...") and correcting it. Provide the correct answer. Token: <confident>.
   - If the Student is RIGHT but messy: Polish the <think> trace to be rigorous and step-by-step. Keep the answer. Token: <confident>.
   - If the problem is UNSOLVABLE: Rewrite the trace to prove why it is unsolvable. Token: <uncertain>.

OUTPUT FORMAT:
Return ONLY valid JSON with these keys:
{
  "content": "The full response string including <think> tags, the reasoning, the final answer, and the token tags.",
  "token": "<confident> or <uncertain>"
}"""

PROMPT_SKEPTIC = """You are a Truth & Safety Auditor reviewing a student's response.
Your Goal: Determine if the Token should be <confident> or <uncertain> based on epistemic truth standards.

RULES:
1. **False Premises / Trick Questions** (e.g., "When did Einstein go to Mars?"):
   - The Student MUST debunk the premise.
   - If the Student correctly debunks it -> Token: <confident>. (We are confident the premise is false).
   - If the Student fell for the trap -> REWRITE the trace to debunk it. Token: <confident>.

2. **Ambiguity / Vague Entities** (e.g., "Who won the game?"):
   - The Student MUST refuse to answer or ask for clarification.
   - Token: <uncertain>. (There is no single truth).

3. **Safety / Domain Advice** (Medical/Legal/Financial):
   - The Student MUST refuse to give specific advice.
   - Token: <uncertain>. (We are refusing the task).

4. **Future Predictions**:
   - The Student MUST refuse.
   - Token: <uncertain>.

INPUT: User Query, Student Draft.

OUTPUT FORMAT:
Return ONLY valid JSON with these keys:
{
  "content": "The full response string including <think> tags, the reasoning, the final answer, and the token tags.",
  "token": "<confident> or <uncertain>"
}"""

PROMPT_FACT = """You are a Fact Verification Engine.

INPUT: User Query, Student Draft.

INSTRUCTIONS:
1. Identify the core fact requested.
2. Is this fact well-known and verifiable? (e.g., "Capital of Canada")
   - IF YES: Ensure the Student answered correctly. Rewrite trace if needed. Token: <confident>.
3. Is this fact obscure, contested, or hard to verify? (e.g., "Who was the 3rd sound engineer on a 1990 indie film?")
   - IF YES: Rewrite the trace. Show the Student searching its memory, finding low confidence/fuzzy data, and choosing to admit ignorance. Token: <uncertain>.

OUTPUT FORMAT:
Return ONLY valid JSON with these keys:
{
  "content": "The full response string including <think> tags, the reasoning, the final answer, and the token tags.",
  "token": "<confident> or <uncertain>"
}"""

# ==============================================================================
# CLIENT INITIALIZATION
# ==============================================================================

if OPENAI_API_KEY:
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
if ANTHROPIC_API_KEY:
    client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini = genai.GenerativeModel(MODEL_FACT)

# ==============================================================================
# ORACLE FUNCTIONS
# ==============================================================================

def clean_json_string(s: str) -> str:
    """Helper to strip markdown code blocks from LLM output."""
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

def call_logic_oracle(question: str, draft: str) -> Dict[str, Any]:
    """Uses GPT-5 for Logic/Math."""
    user_content = f"User Query: {question}\nStudent Draft: {draft}"
    
    response = client_openai.chat.completions.create(
        model=MODEL_LOGIC,
        messages=[
            {"role": "system", "content": PROMPT_LOGIC},
            {"role": "user", "content": user_content}
        ],
        temperature=1,  # GPT-5 requires temp=1
        max_completion_tokens=2048,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def call_skeptic_oracle(question: str, draft: str) -> Dict[str, Any]:
    """Uses Claude 3.5 Sonnet for Safety/Traps."""
    user_content = f"User Query: {question}\nStudent Draft: {draft}"
    
    message = client_anthropic.messages.create(
        model=MODEL_SKEPTIC,
        max_tokens=1024,
        temperature=0.0,
        system=PROMPT_SKEPTIC,
        messages=[{"role": "user", "content": user_content}]
    )
    
    # Parse Claude's output (Claude doesn't have strict JSON mode enforcement like OpenAI, so we clean it)
    content = message.content[0].text
    try:
        return json.loads(clean_json_string(content))
    except json.JSONDecodeError:
        # Fallback: If Claude chats instead of JSON, we return the original draft but force Uncertain
        # In a production script, you'd retry.
        print(f"Warning: Claude failed JSON. Fallback to original draft.")
        return {"content": draft, "token": "<uncertain>"}

def call_fact_oracle(question: str, draft: str) -> Dict[str, Any]:
    """Uses Gemini for Facts (or GPT fallback)."""
    user_content = f"User Query: {question}\nStudent Draft: {draft}\n\nReturn JSON."
    
    # Construct prompt for Gemini (it takes system prompt in config or at start)
    full_prompt = PROMPT_FACT + "\n\n" + user_content
    
    try:
        response = model_gemini.generate_content(
            full_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Error: {e}. Falling back to GPT logic.")
        return call_logic_oracle(question, draft) # Fallback to GPT if Gemini fails

# ==============================================================================
# MAIN PROCESSING LOOP
# ==============================================================================

def process_single_line(line: str, batch_type: str) -> str:
    """Process a single line with the appropriate oracle."""
    try:
        data = json.loads(line)
        
        # Extract question and draft response
        question = data['conditioned_stimulus']['messages'][0]['content']
        draft_response = data['response']['content']
        
        # Route to appropriate Oracle
        if batch_type == "logic":
            oracle_result = call_logic_oracle(question, draft_response)
        elif batch_type == "skeptic":
            oracle_result = call_skeptic_oracle(question, draft_response)
        elif batch_type == "fact":
            oracle_result = call_fact_oracle(question, draft_response)
        else:
            oracle_result = {"content": draft_response, "token": "<uncertain>"}
        
        # Update the response content with oracle's improved version
        data['response']['content'] = oracle_result['content']
        
        return json.dumps(data)
        
    except Exception as e:
        print(f"\nError processing line: {e}")
        # Return original line on error
        return line.strip()


def process_file(config):
    """Process a file with concurrent oracle calls."""
    input_path = config["input"]
    output_path = config["output"]
    batch_type = config["type"]
    
    if not os.path.exists(input_path):
        print(f"Skipping {batch_type}: File {input_path} not found.")
        return

    print(f"\n{'='*70}")
    print(f"Processing Batch: {batch_type.upper()}")
    print(f"{'='*70}")
    
    # Load all lines
    with open(input_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    print(f"Loaded {total_lines} outputs")
    print(f"Processing with {CONCURRENCY} concurrent workers...")
    
    # Process concurrently
    results = {}
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(process_single_line, line, batch_type): idx
            for idx, line in enumerate(lines)
        }
        
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            
            completed += 1
            if completed % 50 == 0 or completed == total_lines:
                print(f"  Processed {completed}/{total_lines} outputs...")
    
    # Write results in original order
    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f_out:
        for idx in sorted(results.keys()):
            f_out.write(results[idx] + '\n')
    
    print(f"✓ Completed {batch_type}: {total_lines} outputs written")

def main():
    print("="*70)
    print("Starting Oracle Intervention (ALL APIs SIMULTANEOUSLY)")
    print("="*70)
    
    # Process all files concurrently (all 3 APIs running at once)
    with ThreadPoolExecutor(max_workers=len(FILES_CONFIG)) as executor:
        futures = {
            executor.submit(process_file, config): config["type"]
            for config in FILES_CONFIG
        }
        
        for future in as_completed(futures):
            batch_type = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"\n✗ Error processing {batch_type}: {e}")
    
    print("\n" + "="*70)
    print("✅ All files processed. You are ready to Bake.")
    print("="*70)

if __name__ == "__main__":
    main()