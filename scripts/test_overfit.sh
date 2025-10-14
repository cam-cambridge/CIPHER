#!/bin/bash

# Default values (matching the Python script defaults)
TEST_SAMPLES=100
MODEL_PATH="cemag/cipher_printing"
QUESTION_ANSWER_PATH="squad"
IMAGE_CAPTION_PATH="nlphuji/flickr30k"
RESULTS_PATH="./results"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test_samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --question_answer_path)
            QUESTION_ANSWER_PATH="$2"
            shift 2
            ;;
        --image_caption_path)
            IMAGE_CAPTION_PATH="$2"
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
            echo "  --test_samples N        Number of test samples (default: 100)"
            echo "  --model_path PATH       Model path (default: cemag/cipher_printing)"
            echo "  --question_answer_path PATH         Question-answer dataset path (default: rajpurkar/squad)"
            echo "  --image_caption_path PATH         Image caption dataset path (default: nlphuji/flickr30k)"
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

echo "Running overfit tests with:"
echo "  Test samples: $TEST_SAMPLES"
echo "  Model path: $MODEL_PATH"
echo "  Question answer path: $QUESTION_ANSWER_PATH"
echo "  Image caption path: $IMAGE_CAPTION_PATH"
echo "  Results path: $RESULTS_PATH"
echo ""

python ./src/tests/examine_overfit.py \
    --test_samples $TEST_SAMPLES \
    --model_path "$MODEL_PATH" \
    --question_answer_path "$QUESTION_ANSWER_PATH" \
    --image_caption_path "$IMAGE_CAPTION_PATH" \
    --results_path "$RESULTS_PATH"

