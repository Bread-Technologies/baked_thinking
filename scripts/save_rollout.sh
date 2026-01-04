#!/bin/bash
# Save bgit rollout output to a file
# Usage (from baked_thinking/baked_thinking/ directory): 
#   ../scripts/save_rollout.sh                    # Save to results/rollout_output.json
#   ../scripts/save_rollout.sh custom_name.json   # Save to results/custom_name.json

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
BAKED_DIR="$PROJECT_ROOT/baked_thinking"
RESULTS_DIR="$PROJECT_ROOT/results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Determine output filename
if [ -z "$1" ]; then
    # Default filename with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_FILE="$RESULTS_DIR/rollout_output_${TIMESTAMP}.json"
else
    # User-provided filename
    OUTPUT_FILE="$RESULTS_DIR/$1"
fi

# Change to baked_thinking subdirectory (where bgit should be run)
cd "$BAKED_DIR" || {
    echo "❌ Error: Could not find baked_thinking directory at $BAKED_DIR"
    exit 1
}

echo "🔍 Fetching rollout output..."
echo "📁 Working directory: $(pwd)"
echo "📁 Saving to: $OUTPUT_FILE"
echo ""

# Run bgit rollout and save output
bgit rollout --limit 3600> "$OUTPUT_FILE" 2>&1

# Check if successful
if [ $? -eq 0 ]; then
    # Count lines in output
    LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
    echo ""
    echo "✅ Rollout output saved successfully!"
    echo "📊 Total lines: $LINE_COUNT"
    echo "📄 File: $OUTPUT_FILE"
else
    echo ""
    echo "❌ Error: Failed to fetch rollout output"
    echo "💡 Make sure you've run 'bgit run rollout' first and it has completed"
    exit 1
fi

