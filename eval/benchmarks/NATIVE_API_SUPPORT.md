# Native API Support

The composite benchmark now supports **native APIs** for Claude, GPT, and Gemini - no routing through Bread API.

## API Routing

| Model | Provider | API Used | Model ID |
|-------|----------|----------|----------|
| `claude-4.5-sonnet` | Anthropic | Anthropic SDK | claude-sonnet-4-20250514 |
| `gpt-5` | OpenAI | OpenAI SDK | gpt-5 |
| `gemini-3.0` | Google | Google Generative AI | gemini-2.0-flash-thinking-exp-01-21 |

## Setup

### 1. Install Native SDKs

```bash
pip install anthropic google-generativeai
```

Or:

```bash
pip install -r requirements.txt
```

### 2. Add API Keys

Edit `external_benchmarks/.env`:

```bash
# Model APIs (native)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=...
```

## Usage

Same simple command - API routing happens automatically:

```bash
# Uses Anthropic native API
python eval/benchmark_composite.py --model claude-4.5-sonnet

# Uses OpenAI native API
python eval/benchmark_composite.py --model gpt-5

# Uses Google native API  
python eval/benchmark_composite.py --model gemini-3.0
```

## Temperature Settings

**Model Inference** (generating responses):
- Temperature: **1** (all models)
- Max tokens: 2048

**Judge Models** (grading responses):
- Temperature: **0** (deterministic)
- GPT-4-turbo: max_tokens=10
- GPT-4o: max_tokens=100

## API Format Normalization

Each provider has different API formats. The script handles this automatically:

### Anthropic API
```python
client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "..."}],
    system="...",  # Separate system parameter
    temperature=1,
    max_tokens=2048
)
```

### OpenAI API
```python
client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "..."}],
    temperature=1,
    max_completion_tokens=2048  # Note: GPT-5 uses max_completion_tokens
)
```

### Google Gemini API
```python
model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
model.generate_content(
    "User: ...\n\n",  # Combined prompt format
    generation_config={
        "temperature": 1,
        "max_output_tokens": 2048
    }
)
```

All wrapped to return consistent string responses.

## Benefits

1. **Native APIs**: Direct access, no middleman
2. **Better reliability**: Provider-specific optimizations
3. **Model-specific features**: Access latest model versions
4. **Automatic routing**: Same simple command interface
5. **Backward compatible**: Can still use Bread API if needed

## Error Handling

If a native SDK isn't installed:

```
ImportError: anthropic package not installed. Run: pip install anthropic
```

**Solution**: Install the required package

If API key is missing:

```
Error: ANTHROPIC_API_KEY not set
```

**Solution**: Add your API key to `external_benchmarks/.env`

## Judge Models

Judge models **always use OpenAI API** (standardized):
- SimpleQA judge: `gpt-4-turbo-preview`
- AbstentionBench judge: `gpt-4o`
- TRIDENT judge: `gpt-4o`

This ensures consistent grading across all model evaluations.


