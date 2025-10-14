#!/bin/bash

# Default values (matching the Python script defaults)
MODEL_PATH="cemag/cipher_printing"
QUESTIONS_PATH="prompts/3d_printing_questions.json"
FACTS_PATH="src/RAG/processed_facts_openai.json"
RAG="False"
RESULTS_PATH="./results"
OPENAI_API_KEY="...."

# Check if API key is empty, not set, or equals "..."
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
        --questions_path)
            QUESTIONS_PATH="$2"
            shift 2
            ;;
        --rag)
            RAG="$2"
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
            echo "  --questions_path PATH     Questions path (default: questions.txt)"
            echo "  --rag BOOL             RAG (default: True)"
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
echo "  Questions path: $QUESTIONS_PATH"
echo "  RAG: $RAG"
echo "  Results path: $RESULTS_PATH"
echo ""

python ./src/tests/domain_expertise.py \
    --model_path "$MODEL_PATH" \
    --questions_path "$QUESTIONS_PATH" \
    --rag "$RAG" \
    --results_path "$RESULTS_PATH"

