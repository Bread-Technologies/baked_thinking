# baked_thinking

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Understanding the Files](#understanding-the-files)
- [Using bgit Commands](#using-bgit-commands)
- [Branching and Merging](#branching-and-merging)
- [Workflow Examples](#workflow-examples)
- [Understanding the Workflow](#understanding-the-workflow)
- [Repository Structure](#repository-structure)
- [Tips and Best Practices](#tips-and-best-practices)

---

## Overview

This repository uses `bgit` (Bread Git Interface), a git-native version control system for AI model baking. `bgit` integrates seamlessly with Git workflows, treating model configurations as code and tracking model lineage through commits and branches.

### Core Philosophy

- **`input.yml`**: Think of this as an input box. You write your configuration here, and after successful bake operations, it gets cleared (keeping only section headings) to prepare for the next iteration.
- **`recipe.yml`**: Contains the history and lineage of your model up to the current commit. It tracks the bakes that have led to this point, their model names, and the sequential relationship between models. This file is automatically updated after each successful bake and committed to Git. To see bakes across all branches, use `bgit tree`.
- **`.bread`**: Internal state file that maps your YAML configuration to Bread SDK resources. It stores the repository name (set during `bgit init`), tracks prompts, targets, bakes, and the parent model for sequential bakes. This file is auto-generated and should not be edited manually.

---

## Getting Started

### 1. Configure Your Model

Open `input.yml` and customize your configuration:

- **PROMPT**: Teacher and student prompts (teacher generates training data, student is used at inference)
- **TARGET**: Training data generation configuration (generators, questions)
- **BAKE**: Model training settings (datasets, trajectories)

**Note**: The repository name is set during `bgit init` and stored in the `.bread` file. You don't need to configure it in `input.yml`.

The `input.yml` file includes documentation links and comments to guide you.

### 2. Stage and Commit Your Configuration

**Important**: `bgit` only reads from `input.yml` when changes are staged (via `git add`). This ensures you're working with committed configurations:

```bash
# Stage your input.yml changes
bgit add input.yml

# Commit your configuration
bgit commit -m "Initial model configuration: Yoda personality"
```

The `pre-commit` hook automatically updates the `.bread` file when YAML files are staged.

### 3. Run Your First Bake

```bash
# Run all operations in sequence
bgit run stim rollout bake
```

This will:
1. **stim**: Generate questions using your target generators
2. **rollout**: Generate responses to those questions using your model
3. **bake**: Train/fine-tune your model on the generated data

After a successful bake:
- `recipe.yml` is updated with the new bake and model name
- `input.yml` is cleared (keeping only section headings) for your next iteration
- `.bread` file is updated with the new model as `PARENT_MODEL` for sequential bakes

---

## Understanding the Files

### `input.yml` - Your Configuration Input Box

This is where you write your model configuration. Think of it as a form you fill out:

- **Purpose**: Define prompts, targets, and bake settings
- **Behavior**: After successful bake operations, the content is cleared (except headings) to prepare for the next iteration
- **Staging Requirement**: `bgit run` only reads from staged changes. Always `bgit add input.yml` before running operations
- **Documentation**: Includes links to Bread documentation for each section

### `recipe.yml` - Model History and Lineage

This file tracks the history of your model up to the current commit:

- **Purpose**: Records the bakes that have led to this commit, their names, and the resulting model names
- **Structure**: Contains a `resources.bakes` array with each bake's name and model names
- **Lineage**: Shows the sequential relationship between models (parent → child) in the current branch/commit
- **Per-commit**: Each commit has its own version of `recipe.yml` reflecting the bakes up to that point
- **Auto-updated**: Automatically updated and committed after each successful bake
- **Read-only**: Has a header indicating it's auto-generated (though you can view it)
- **Cross-branch view**: Use `bgit tree` to see bakes across all branches by reading `recipe.yml` from each branch

Example structure:
```yaml
resources:
  bakes:
    - bake_name: bake1_abc123
      model_names:
        - user/repo/bake1_abc123/120
    - bake_name: bake2_def456
      model_names:
        - user/repo/bake2_def456/150
```

### `.bread` - Internal State File

This file maintains the mapping between your YAML configuration and Bread SDK resources:

- **Purpose**: Tracks prompts, targets, bakes, and their corresponding names/IDs on the Bread platform
- **Auto-generated**: Created and updated automatically by `bgit`
- **Never edit manually**: Contains hashes and internal state
- **PARENT_MODEL**: Stores the model name from the last successful bake for sequential bakes

---

## Using bgit Commands

### Basic Git Operations

`bgit` provides wrappers for common git commands:

```bash
# Stage files (triggers pre-commit hook to update .bread)
bgit add input.yml

# Commit changes
bgit commit -m "Your commit message"

# All other git commands work normally
git push origin main
git pull
git checkout -b branch-name
```

### `bgit init <model-name>`

Initialize a new Bread model repository. Creates:
- Directory structure
- `input.yml` template with documentation
- `.bread` file
- `recipe.yml` file
- Git repository with hooks
- `.gitattributes` for merge behavior

### `bgit status`

View the status of your stim, rollout, and bake operations:

```bash
bgit status
```

Shows:
- **STIM**: Status, progress percentage, and line count (when complete)
- **ROLLOUT**: Status, progress percentage, and line count (when complete)
- **BAKE**: Status and progress percentage

Example output:
```
✓ target1: complete (100%) - 1500 lines
✓ target1: complete (100%) - 7500 lines
⏳ bake1: running (45%)
```

### `bgit run <operations...>`

Run one or more operations: `stim`, `rollout`, or `bake`:

```bash
# Run individual operations
bgit run stim
bgit run rollout
bgit run bake

# Run multiple operations in sequence
bgit run stim rollout bake
```

**Important Notes**:
- **Staging Required**: For `bake` operations, all changes must be staged. `bgit run bake` will fail if there are unstaged changes.
- **Sequential Bakes**: When running `bake`, the `PARENT_MODEL` from `.bread` is automatically used. You cannot override this in `input.yml`.
- **YAML Clearing**: After successful bake operations, `input.yml` is cleared (keeping headings) to prepare for the next iteration.
- **Detached Mode**: Use `-d` or `--detach` flag to run bake operations in the background (useful for long-running bakes that might conflict with Git state changes).
- **Bake Completion**: When a bake completes successfully, the terminal displays:
  - Model name(s) created by the bake
  - Loss functions (latest_loss, final_loss, min_loss, max_loss) if available

### `bgit stim` and `bgit rollout`

View outputs from completed operations:

```bash
# View stim outputs
bgit stim

# View stim outputs with limit
bgit stim --limit 10

# View stim count only
bgit stim --count

# View rollout outputs
bgit rollout

# View rollout outputs with limit
bgit rollout --limit 10

# View rollout count only
bgit rollout --count
```

### `bgit models`

List all available models on your Bread account:

```bash
bgit models
```

Shows model names

### `bgit target`

Target management commands:

```bash
# List all targets
bgit target ls
# or
bgit target list

# Get detailed information about a specific target
bgit target <target-name>
```

Shows target configuration, generators, and other metadata.

### `bgit tree`

Display a visual tree of your model lineage across all branches:

```bash
bgit tree
```

This command:
- Reads `recipe.yml` from all git branches (each branch's `recipe.yml` contains bakes up to that branch's commits)
- Builds a tree showing parent-child relationships between models across branches
- Displays branch information for each bake
- Shows model names in a hierarchical structure
- Combines the per-commit `recipe.yml` files to show the complete picture across all branches

The tree uses colors to distinguish:
- **Yellow**: Bake names
- **Green**: Model names
- **Gray**: Branch labels
- **Cyan**: Branch headers

Example output:
```
🌳 Model Tree for my_repo (across all branches)

Base Model: Qwen/Qwen3-32B

└── bake1_abc123
    └─ user/repo/bake1_abc123/120
    └── bake2_def456 [main]
        └─ user/repo/bake2_def456/150
```

This is especially useful for understanding model relationships when working across multiple branches.

### `bgit chat <model-name>`

Start an interactive chat session with a baked model:

```bash
bgit chat user/repo/bake_name/model_id
```

Supports commands:
- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/system <prompt>` - Set system prompt
- `/status` - Show current settings
- `/quit` - Exit chat

### `bgit fetch`

Fetch results from detached bake operations (when using `bgit run bake --detach`):

```bash
bgit fetch
```

---

## Branching and Merging

### Creating Feature Branches

```bash
# Create and switch to a new branch
git checkout -b experiment/new-idea

# Edit input.yml
vim input.yml

# Stage, commit, and run operations
bgit add input.yml
bgit commit -m "Experiment: increased temperature for creativity"
bgit run stim rollout bake

# Push branch
git push -u origin experiment/new-idea
```

### Merging Branches

When merging a feature branch into `main`, the merge strategy ensures that `.bread` and `recipe.yml` from the incoming branch overwrite the versions on `main`:

**How it works**:
1. `.gitattributes` configures a custom merge driver (`keepTheirs`) for `.bread` and `recipe.yml`
2. During merge, Git automatically takes the incoming branch's version of these files
3. A `post-merge` hook acts as a safety net, ensuring these files use the incoming branch's version even if the merge driver didn't run

**Why this behavior?**
- Feature branches represent complete model experiments
- When merging, you want the feature branch's model state (`.bread`) and history (`recipe.yml`) to become the new main state
- This ensures `main` always reflects the latest successful model configuration

**Example merge workflow**:
```bash
# On main branch
git checkout main

# Merge feature branch
git merge experiment/new-idea

# .bread and recipe.yml are automatically overwritten with feature branch's versions
# Continue working or run new operations
bgit run stim rollout bake
```

---

## Workflow Examples

### Iterating on Your Model

```bash
# 1. Edit input.yml with your changes
vim input.yml

# 2. Stage your changes (required for bgit to read them)
bgit add input.yml

# 3. Commit with a descriptive message
bgit commit -m "Increased temperature to 0.8 for more creative responses"

# 4. Run operations
bgit run stim rollout bake

# 5. Check status
bgit status

# 6. View outputs if needed
bgit stim --count
bgit rollout --count
```

After a successful bake, `input.yml` is cleared and ready for your next iteration.

### Sequential Bakes (Building on Previous Models)

When you run multiple bakes sequentially, each bake uses the previous bake's model as the parent:

```bash
# First bake
bgit add input.yml
bgit commit -m "Initial bake"
bgit run stim rollout bake
# Creates model: user/repo/bake1_abc123/120

# Second bake (automatically uses bake1_abc123/120 as parent)
# Edit input.yml (PARENT_MODEL is automatically set from .bread)
bgit add input.yml
bgit commit -m "Second iteration"
bgit run stim rollout bake
# Creates model: user/repo/bake2_def456/150 (parent: bake1_abc123/120)
```

The `PARENT_MODEL` is stored in `.bread` and automatically used for sequential bakes. You can override it for `TARGET` operations but not for `BAKE` operations.

### Experimenting with Branches

```bash
# Create experiment branch
git checkout -b experiment/higher-temperature

# Edit input.yml
vim input.yml  # Change temperature settings

# Stage, commit, and run
bgit add input.yml
bgit commit -m "Experiment: higher temperature for more variation"
bgit run stim rollout bake

# If experiment is successful, merge back to main
git checkout main
git merge experiment/higher-temperature
# .bread and recipe.yml from experiment branch overwrite main's versions

# Continue with merged state
bgit run stim rollout bake
```

### Long-Running Bakes with Detached Mode

For bakes that take a long time and might conflict with Git state changes:

```bash
# Run bake in detached mode
bgit run bake --detach

# Continue working on other branches or make commits
git checkout -b other-work
# ... make changes ...

# Later, fetch results from detached bake
bgit fetch
```

---

## Understanding the Workflow

### The Complete Flow

```
1. Edit input.yml          → Write your configuration
2. bgit add input.yml      → Stage changes (bgit reads from staged files)
3. bgit commit -m "..."    → Commit configuration
4. bgit run stim           → Generate questions
5. bgit run rollout        → Generate responses
6. bgit run bake           → Train model
   ├─ Updates recipe.yml   → Records bake in history
   ├─ Updates .bread       → Sets PARENT_MODEL for next bake
   ├─ Clears input.yml     → Prepares for next iteration
   └─ Displays results     → Shows model name(s) and loss functions
7. bgit status             → Check operation status
8. bgit stim/rollout       → View outputs
9. bgit tree               → Visualize model lineage across branches
```

### Key Behaviors

- **Staging Requirement**: `bgit run` (especially `bake`) requires changes to be staged. This ensures you're working with committed configurations.
- **Automatic Clearing**: `input.yml` is cleared after successful bakes, keeping only section headings and documentation comments.
- **Model Lineage**: Each bake automatically uses the previous bake's model as the parent (stored in `.bread` as `PARENT_MODEL`).
- **Merge Behavior**: When merging branches, `.bread` and `recipe.yml` from the incoming branch overwrite the current branch's versions.
- **Recipe History**: `recipe.yml` maintains the history of bakes up to the current commit. Each commit contains the `recipe.yml` state at that point in the Git history.

---

## Repository Structure

### How Bake Commits Layer on Git

`bgit` is designed to work seamlessly with Git's existing commit history. Bake operations create commits that layer on top of your existing Git tree, preserving the full history of your model development.

**Commit Structure:**

Each bake operation creates a commit that includes:
- Updated `recipe.yml` - Records the new bake and model names
- Updated `.bread` - Updates `PARENT_MODEL` for the next sequential bake
- Cleared `input.yml` - Prepared for the next iteration

**Git History Example:**

```
* commit abc123 (HEAD -> main)
| Update recipe.yml after bake completion: bake3_xyz789
| 
* commit def456
| Update recipe.yml after bake completion: bake2_def456
| 
* commit ghi789
| Update recipe.yml after bake completion: bake1_abc123
| 
* commit jkl012
| Initial model configuration: Yoda personality
```

**Branch-Based Development:**

When working with branches, each branch maintains its own model lineage:

```
main:           bake1 → bake2 → bake3
                 ↓
experiment:     bake1 → bake4 → bake5
                 ↓
feature:        bake1 → bake6
```

When branches are merged, the incoming branch's `.bread` and `recipe.yml` overwrite the current branch's versions, ensuring the merged branch reflects the complete model state from that branch.

**File Tracking:**

- **`input.yml`**: Tracked in Git, cleared after bakes but committed empty (with headings)
- **`recipe.yml`**: Auto-committed after each successful bake, tracks model history up to that commit
- **`.bread`**: Auto-committed via pre-commit hook when YAML changes, tracks SDK resource mappings
- **`.gitattributes`**: Configures merge behavior for `.bread` and `recipe.yml`
- **Git hooks**: `pre-commit` updates `.bread`, `post-merge` ensures correct merge resolution

**Benefits of This Structure:**

1. **Full History**: Every bake is a Git commit, providing complete audit trail
2. **Branch Isolation**: Each branch can develop models independently
3. **Easy Rollback**: Use `git revert` or `git checkout` to go back to any model state
4. **Collaboration**: Standard Git workflows (pull requests, code review) work seamlessly
5. **Lineage Tracking**: Each commit's `recipe.yml` maintains model relationships up to that point. Use `bgit tree` to visualize relationships across all branches.

**Advanced Usage:**

You can use standard Git commands to inspect your model history:

```bash
# See all bake commits
git log --oneline --grep="recipe.yml"

# View recipe.yml at a specific commit
git show abc123:recipe.yml

# Compare recipe.yml between commits
git diff abc123 def456 -- recipe.yml

# See model lineage for a specific branch
git checkout experiment-branch
bgit tree
```

---

## Tips and Best Practices

1. **Always stage before running**: Use `bgit add input.yml` before `bgit run` to ensure your changes are read correctly.

2. **Descriptive commit messages**: Your commit messages document what changed in each model version. Be descriptive!

3. **Check status regularly**: Use `bgit status` to monitor operation progress and see line counts.

4. **Use branches for experiments**: Create feature branches for different model configurations and merge successful ones back to main.

5. **Sequential bakes**: Each bake builds on the previous one. The `PARENT_MODEL` is automatically tracked in `.bread`.

6. **View counts**: Use `bgit stim --count` and `bgit rollout --count` to quickly check how many outputs were generated.

7. **Detached mode for long bakes**: Use `--detach` flag for bakes that might take a long time and could conflict with Git state changes.

8. **Visualize model lineage**: Use `bgit tree` to see all your models across branches in a visual tree structure.

9. **Inspect targets**: Use `bgit target ls` to see all targets, and `bgit target <name>` to get detailed information about a specific target.

---

## Important Notes

### Undoing Bakes

**Use `git revert` only if you want to permanently undo a bake**. This will:
- Remove the bake commit from history
- Revert `recipe.yml` to the previous state
- Revert `.bread` to the previous state

**Warning**: This is a destructive operation. If you've already merged the bake or shared it with others, consider creating a new branch with a different approach instead of reverting.

```bash
# Only use if you're certain you want to undo a bake
git revert <bake-commit-hash>
```

**Alternative**: Instead of reverting, create a new branch to try a different approach. This preserves history and allows you to compare results.

---

*This README is auto-generated during init. Update it as your model evolves!*
