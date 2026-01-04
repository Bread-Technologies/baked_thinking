# Scripts for Saving bgit Outputs

These scripts help you save stim and rollout outputs from bgit to files in the `results/` directory.

**Important:** Run these scripts from the `baked_thinking/baked_thinking/` directory where your `input.yml` is located.

## Available Scripts

### 1. `save_stim.sh` - Save Stim Output

Saves the generated questions from `bgit stim` to a JSON file.

**Usage (from `baked_thinking/baked_thinking/` directory):**
```bash
# Save with automatic timestamp
../scripts/save_stim.sh

# Save with custom filename
../scripts/save_stim.sh my_stim_v1.json
```

**Output:** `results/stim_output_YYYYMMDD_HHMMSS.json` or `results/[custom_name].json`

---

### 2. `save_rollout.sh` - Save Rollout Output

Saves the model responses from `bgit rollout` to a JSON file.

**Usage (from `baked_thinking/baked_thinking/` directory):**
```bash
# Save with automatic timestamp
../scripts/save_rollout.sh

# Save with custom filename
../scripts/save_rollout.sh my_rollout_v1.json
```

**Output:** `results/rollout_output_YYYYMMDD_HHMMSS.json` or `results/[custom_name].json`

---

### 3. `save_outputs.sh` - Save Both Stim and Rollout

Saves both stim and rollout outputs at once.

**Usage (from `baked_thinking/baked_thinking/` directory):**
```bash
# Save both with automatic timestamp
../scripts/save_outputs.sh

# Save both with custom base name
../scripts/save_outputs.sh experiment_v2
```

**Output:**
- `results/YYYYMMDD_HHMMSS_stim.json` and `results/YYYYMMDD_HHMMSS_rollout.json`
- OR `results/experiment_v2_stim.json` and `results/experiment_v2_rollout.json`

---

## Workflow Example

```bash
# Make sure you're in the baked_thinking/baked_thinking/ directory
cd baked_thinking

# 1. Run stim and rollout
bgit run stim rollout

# 2. Check status (wait for completion)
bgit status

# 3. Save outputs when complete
../scripts/save_outputs.sh metacognitive_v2

# Result:
# ✅ results/metacognitive_v2_stim.json
# ✅ results/metacognitive_v2_rollout.json
```

---

## Prerequisites

- Run scripts from the `baked_thinking/baked_thinking/` directory (where `input.yml` is)
- Make sure you've run `bgit run stim` or `bgit run rollout` first
- Wait for the operations to complete (check with `bgit status`)
- Scripts will save to the `results/` directory (created automatically)

---

## Directory Structure

```
baked_thinking/               # Project root
├── baked_thinking/           # Working directory (run scripts from here!)
│   └── input.yml
├── scripts/                  # Scripts location
│   ├── save_stim.sh
│   ├── save_rollout.sh
│   └── save_outputs.sh
└── results/                  # Output files saved here
    ├── metacognitive_v2_stim.json
    └── metacognitive_v2_rollout.json
```

---

## Troubleshooting

**Error: "Failed to fetch output"**
- Make sure the stim/rollout job has completed
- Run `bgit status` to check
- Ensure you're in the `baked_thinking/baked_thinking/` directory

**Error: "Could not find baked_thinking directory"**
- The scripts need to be in the `scripts/` folder at project root
- Make sure you're calling them with `../scripts/save_*.sh`

**Permission denied**
```bash
# From project root
chmod +x scripts/*.sh
```
