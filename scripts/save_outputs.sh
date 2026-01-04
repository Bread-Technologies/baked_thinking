#!/bin/bash
# Save both bgit stim and rollout outputs to files
# Usage (from baked_thinking/baked_thinking/ directory): 
#   ../scripts/save_outputs.sh                    # Save with timestamp
#   ../scripts/save_outputs.sh my_experiment      # Save as my_experiment_stim.json and my_experiment_rollout.json

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
BAKED_DIR="$PROJECT_ROOT/baked_thinking"
RESULTS_DIR="$PROJECT_ROOT/results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Determine base filename
if [ -z "$1" ]; then
    # Default filename with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BASE_NAME="${TIMESTAMP}"
else
    # User-provided base name
    BASE_NAME="$1"
fi

STIM_FILE="$RESULTS_DIR/${BASE_NAME}_stim.json"
ROLLOUT_FILE="$RESULTS_DIR/${BASE_NAME}_rollout.json"

# Change to baked_thinking subdirectory (where bgit should be run)
cd "$BAKED_DIR" || {
    echo "❌ Error: Could not find baked_thinking directory at $BAKED_DIR"
    exit 1
}

echo "🔍 Fetching stim and rollout outputs..."
echo "📁 Working directory: $(pwd)"
echo ""

# Save stim output
echo "📥 Saving stim output..."
bgit stim > "$STIM_FILE" 2>&1
if [ $? -eq 0 ]; then
    STIM_LINES=$(wc -l < "$STIM_FILE")
    echo "   ✅ Stim saved: $STIM_FILE ($STIM_LINES lines)"
else
    echo "   ❌ Failed to fetch stim output"
    STIM_ERROR=true
fi

echo ""

# Save rollout output
echo "📥 Saving rollout output..."
bgit rollout > "$ROLLOUT_FILE" 2>&1
if [ $? -eq 0 ]; then
    ROLLOUT_LINES=$(wc -l < "$ROLLOUT_FILE")
    echo "   ✅ Rollout saved: $ROLLOUT_FILE ($ROLLOUT_LINES lines)"
else
    echo "   ❌ Failed to fetch rollout output"
    ROLLOUT_ERROR=true
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -z "$STIM_ERROR" ] && [ -z "$ROLLOUT_ERROR" ]; then
    echo "✅ Both outputs saved successfully!"
    echo "📁 Location: $RESULTS_DIR/"
    echo "   • ${BASE_NAME}_stim.json"
    echo "   • ${BASE_NAME}_rollout.json"
elif [ -n "$STIM_ERROR" ] && [ -n "$ROLLOUT_ERROR" ]; then
    echo "❌ Failed to fetch both outputs"
    echo "💡 Make sure you've run 'bgit run stim rollout' and both have completed"
    exit 1
else
    echo "⚠️  Partial success - check messages above"
    exit 1
fi

