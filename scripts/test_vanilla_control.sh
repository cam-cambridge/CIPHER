#!/bin/bash

# Default values (matching the Python script defaults)
MODEL_PATH="cemag/cipher_printing"
NUM_QUESTIONS=10
PROMPT_PATH="./prompts/vanilla_control.txt"
RESULTS_PATH="./results"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num_questions)
            NUM_QUESTIONS="$2"
            shift 2
            ;;
        --prompt_path)
            PROMPT_PATH="$2"
            shift 2
            ;;
        --results_path)
            RESULTS_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model_path PATH       Model path (default: cemag/cipher_printing)"
            echo "  --num_questions INT     Number of questions (default: 500)"
            echo "  --prompt_path PATH     Prompt path (default: ./prompts/vanilla_control.txt)"
            echo "  --results_path PATH     Results output path (default: ./results)"
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
echo "  Num questions: $NUM_QUESTIONS"  
echo "  Prompt path: $PROMPT_PATH"
echo "  Results path: $RESULTS_PATH"
echo ""

python ./src/tests/vanilla_control.py \
    --model_path "$MODEL_PATH" \
    --num_questions "$NUM_QUESTIONS" \
    --prompt_path "$PROMPT_PATH" \
    --results_path "$RESULTS_PATH"
