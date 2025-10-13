#!/bin/bash

# Default values (matching the Python script defaults)
TEST_SAMPLES=100
BATCH_SIZE=8
MODEL_PATH="cemag/cipher_printing"
CSV_PATH="./src/data/test.csv"
IMGS_PATH="./src/data/images/"
RESULTS_PATH="./results"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test_samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --csv_path)
            CSV_PATH="$2"
            shift 2
            ;;
        --imgs_path)
            IMGS_PATH="$2"
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
            echo "  --test_samples N        Number of test samples (default: 500)"
            echo "  --batch_size N          Batch size (default: 8)"
            echo "  --model_path PATH       Model path (default: cemag/cipher_printing)"
            echo "  --csv_path PATH         CSV dataset path (default: ./src/data/dataset.csv)"
            echo "  --imgs_path PATH        Images directory path (default: ./src/data/sample_dataset/)"
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

echo "Running flowrate predictions with:"
echo "  Test samples: $TEST_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Model path: $MODEL_PATH"
echo "  CSV path: $CSV_PATH"
echo "  Images path: $IMGS_PATH"
echo "  Results path: $RESULTS_PATH"
echo ""

python ./src/test_utils/predict_flowrates.py \
    --test_samples $TEST_SAMPLES \
    --batch_size $BATCH_SIZE \
    --model_path "$MODEL_PATH" \
    --csv_path "$CSV_PATH" \
    --imgs_path "$IMGS_PATH" \
    --results_path "$RESULTS_PATH"

