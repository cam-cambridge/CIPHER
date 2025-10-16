#!/bin/bash

# Default values (matching the Python script defaults)
MODEL_PATH="cemag/cipher_printing"
RESULTS_PATH="./results"
QUESTION="./prompts/ask.txt"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --question)
            QUESTION="$2"
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
            echo "  --question PATH         Question path (default: prompts/ask.txt)"
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

echo "Running ask with:"
echo "  Model path: $MODEL_PATH"
echo "  Question: $QUESTION"
echo "  Results path: $RESULTS_PATH"
echo ""

python ./src/tests/ask.py \
    --model_path "$MODEL_PATH" \
    --question "$QUESTION" \
    --results_path "$RESULTS_PATH"

