#!/bin/bash

# Default values (matching the Python script defaults)
MODEL_PATH="cemag/cipher_printing"
RESULTS_PATH="./results"
TEMPLATE_PATH="./prompts/geometry_expert.txt"
GEOMETRY_REQUEST="a 20mm dimaeter circle."

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --template_path)
            TEMPLATE_PATH="$2"
            shift 2
            ;;
        --geometry_request)
            GEOMETRY_REQUEST="$2"
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
            echo "  --template_path PATH         Template path (default: prompts/geometry_expert.txt)"
            echo "  --geometry_request PATH         Geometry request (default: a 20mm dimaeter circle.)"
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

echo "Running geometry expert (ask.py proxy) with:"
echo "  Model path: $MODEL_PATH"
echo "  Template path: $TEMPLATE_PATH"
echo "  Geometry request: $GEOMETRY_REQUEST"
echo "  Results path: $RESULTS_PATH"
echo ""

python ./src/tests/geometry_expert.py \
    --model_path "$MODEL_PATH" \
    --template_path "$TEMPLATE_PATH" \
    --geometry_request "$GEOMETRY_REQUEST" \
    --results_path "$RESULTS_PATH"
