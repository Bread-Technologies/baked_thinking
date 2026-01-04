#!/usr/bin/env python3
"""
Interactive testing script for metacognitive teacher prompts.
Chat with Qwen using the teacher prompt to see how it responds with confidence tokens.

Usage:
    python scripts/test_prompt.py
    python scripts/test_prompt.py --prompt prompts/teacher_v2.json
    python scripts/test_prompt.py --base  # Test without teacher prompt (baseline)

Commands during chat:
    /clear    - Clear conversation history
    /base     - Toggle base mode (no teacher prompt)
    /reload   - Reload the prompt file
    /history  - Show conversation history
    /quit     - Exit
"""

import json
import sys
from pathlib import Path
from openai import OpenAI

# Add parent dir to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import BREAD_API_BASE, BREAD_API_KEY, MODEL


def load_prompt(prompt_path: str) -> list[dict]:
    """Load teacher prompt messages from JSON file."""
    with open(prompt_path) as f:
        data = json.load(f)
    return data["messages"]


def highlight_confidence_token(text: str) -> str:
    """Highlight confidence tokens in the response."""
    # ANSI colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    
    text = text.replace("<confident>", f"{BOLD}{GREEN}<confident>{RESET}")
    text = text.replace("<uncertain>", f"{BOLD}{YELLOW}<uncertain>{RESET}")
    
    # Highlight <think> blocks
    if "<think>" in text and "</think>" in text:
        parts = text.split("<think>")
        formatted_parts = [parts[0]]
        for part in parts[1:]:
            if "</think>" in part:
                think_content, rest = part.split("</think>", 1)
                formatted_parts.append(f"{CYAN}<think>{think_content}</think>{RESET}{rest}")
            else:
                formatted_parts.append(f"<think>{part}")
        text = "".join(formatted_parts)
        
    return text


def print_response(content: str):
    """Print assistant response with highlighted tokens."""
    print(f"\nAssistant:\n{highlight_confidence_token(content)}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test metacognitive teacher prompts")
    parser.add_argument("--prompt", default="prompts/teacher_v3_thinking.json", help="Path to prompt JSON")
    parser.add_argument("--base", action="store_true", help="Run without teacher prompt (baseline)")
    args = parser.parse_args()
    
    # Initialize client
    client = OpenAI(base_url=BREAD_API_BASE, api_key=BREAD_API_KEY)
    
    # Load teacher prompt
    prompt_path = Path(__file__).parent.parent / args.prompt
    teacher_messages = [] if args.base else load_prompt(prompt_path)
    
    # Conversation state
    conversation = []
    base_mode = args.base
    
    print("=" * 60)
    print("Metacognitive Prompt Tester")
    print(f"Model: {MODEL}")
    print(f"Mode: {'BASE (no teacher prompt)' if base_mode else f'TEACHER ({args.prompt})'}")
    print("Commands: /clear /base /reload /history /quit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
            
        if not user_input:
            continue
            
        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            
            if cmd == "/quit":
                print("Goodbye!")
                break
                
            elif cmd == "/clear":
                conversation = []
                print("Conversation cleared.")
                continue
                
            elif cmd == "/base":
                base_mode = not base_mode
                conversation = []  # Clear on mode switch
                mode_str = "BASE (no teacher prompt)" if base_mode else "TEACHER"
                print(f"Switched to {mode_str} mode. Conversation cleared.")
                continue
                
            elif cmd == "/reload":
                teacher_messages = load_prompt(prompt_path)
                print(f"Reloaded prompt from {prompt_path}")
                continue
                
            elif cmd == "/history":
                if not conversation:
                    print("No conversation history.")
                else:
                    for msg in conversation:
                        role = msg["role"].upper()
                        print(f"\n[{role}]: {msg['content'][:200]}...")
                continue
                
            else:
                print(f"Unknown command: {cmd}")
                continue
        
        # Build messages for API call
        conversation.append({"role": "user", "content": user_input})
        
        if base_mode:
            messages = conversation.copy()
        else:
            messages = teacher_messages + conversation
        
        # Call API
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=5000,
            )
            
            assistant_content = response.choices[0].message.content
            conversation.append({"role": "assistant", "content": assistant_content})
            
            print_response(assistant_content)
            
            # Quick token check
            has_confident = "<confident>" in assistant_content.lower()
            has_uncertain = "<uncertain>" in assistant_content.lower()
            if not has_confident and not has_uncertain:
                print("⚠️  No confidence token detected in response")
                
        except Exception as e:
            print(f"\nError: {e}")
            conversation.pop()  # Remove failed user message


if __name__ == "__main__":
    main()

