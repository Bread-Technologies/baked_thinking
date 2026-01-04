# Long Context Bgit Directory

## Important Note

The `long_context_bgit` directory contains a bgit (Bread Git Interface) repository with its own git history that is essential for its functionality.

## Extraction Instructions

To extract and use the `long_context_bgit` directory with its git repository intact:

```bash
# Extract the archive
tar -xzf long_context_bgit_with_git.tar.gz

# Verify the git repository is intact
cd long_context_bgit
git status
git log --oneline -5
```

## Why This Approach?

Git doesn't normally allow nesting git repositories within each other. To preserve the `long_context_bgit` directory with its critical `.git` folder intact, we've archived it. This ensures:

1. The complete git history is preserved
2. All bgit functionality remains operational
3. The repository can be used immediately after extraction

## Directory Contents

The `long_context_bgit` directory contains:
- `.git/` - Git repository (essential for bgit functionality)
- `.bread` - Bread configuration file
- `input.yml` - Input configuration for bgit
- `recipe.yml` - Recipe configuration showing bake history
- `README.md` - Original bgit documentation
- Other bgit-related files and templates

## Using the Directory

After extraction, you can use all bgit commands normally:

```bash
cd long_context_bgit
bgit status
bgit tree
# etc.
```

The git history shows the development of baked models and is integral to the bgit workflow.