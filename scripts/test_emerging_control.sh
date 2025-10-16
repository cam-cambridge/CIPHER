#!/bin/bash

# Default values (matching the Python script defaults)
MODEL_PATH="cemag/cipher_printing"
INSTRUCTIONS_PATH="./prompts/emerging_control.txt"

SCENARIOS_PATH="./prompts/unknown_scenarios.json"
NUM_QUESTIONS=10

RAG=true
CONTEXT=5
OPENAI_API_KEY="..."
CODEBOOK=true

RESULTS_PATH="./results"

# This will need an OpenAI API key
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "..." ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set or contains placeholder value"
    exit 1
fi

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --rag)
            RAG=true
            shift 1
            ;;
        --codebook)
            CODEBOOK=true
            shift 1
            ;;
        --instructions_path)
            INSTRUCTIONS_PATH="$2"
            shift 2
            ;;
        --scenarios_path)
            SCENARIOS_PATH="$2"
            shift 2
            ;;
        --results_path)
            RESULTS_PATH="$2"
            shift 2
            ;;
        --context)
            CONTEXT="$2"
            shift 2
            ;;
        --num_questions)
            NUM_QUESTIONS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_path PATH       Model path (default: cemag/cipher_printing)"
            echo "  --rag BOOL             RAG (default: False)"
            echo "  --codebook BOOL        Codebook (default: False)"
            echo "  --instructions_path PATH     Instructions path (default: prompts/emergent_control.txt)"
            echo "  --scenarios_path PATH     Scenarios path (default: prompts/unknown_scenarios.json)"
            echo "  --results_path PATH     Results output path (default: ./results)"
            echo "  --context INT           Number of context facts (default: 5)"
            echo "  --num_questions INT     Number of questions (default: 10)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

export OPENAI_API_KEY="$OPENAI_API_KEY"

echo "Running ask with:"
echo "  Model path: $MODEL_PATH"
echo "  Instructions path: $INSTRUCTIONS_PATH"
echo "  Scenarios path: $SCENARIOS_PATH"
echo "  Num questions: $NUM_QUESTIONS"

echo "  RAG: $RAG"
echo "  Context: $CONTEXT"
echo "  Codebook: $CODEBOOK"

echo "  Results path: $RESULTS_PATH"
echo ""

# Build the command with conditional boolean flags
CMD="python ./src/tests/emerging_control.py \
    --model_path \"$MODEL_PATH\" \
    --instructions_path \"$INSTRUCTIONS_PATH\" \
    --scenarios_path \"$SCENARIOS_PATH\" \
    --num_questions \"$NUM_QUESTIONS\" \
    --context \"$CONTEXT\" \
    --results_path \"$RESULTS_PATH\""

# Add boolean flags only if they are true
if [ "$RAG" = "true" ]; then
    CMD="$CMD --rag"
fi

if [ "$CODEBOOK" = "true" ]; then
    CMD="$CMD --codebook"
fi

# Execute the command
eval $CMD
