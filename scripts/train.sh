#!/bin/bash

# Default values (matching the Python script defaults)
LANGUAGE=False
VISION=False
EXPERT=True
LORA=False
EXP="CIPHER_EXPERIMENT_1"
DATA_PATH="cemag/tl-caxton"
OUTPUT_DIR="./models"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp)
            EXP="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --vision)
            VISION="$2"
            shift 2
            ;;
        --expert)
            EXPERT="$2"
            shift 2
            ;;
        --lora)
            LORA="$2"
            shift 2
            ;;
        --train_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --val_path)
            VAL_PATH="$2"
            shift 2
            ;;
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --exp N        Number of test samples (default: 500)"
            echo "  --language BOOL         Use language model (default: False)"
            echo "  --vision BOOL           Use vision model (default: False)"
            echo "  --expert BOOL           Use expert model (default: True)"
            echo "  --lora BOOL             Use LoRA (default: False)"
            echo "  --dataset PATH       Training data path (default: cemag/tl-caxton)"
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

echo "Running train with:"
echo "  Exp: $EXP"
echo "  Language: $LANGUAGE"
echo "  Vision: $VISION"
echo "  Expert: $EXPERT"
echo "  LoRA: $LORA"
echo "  Data path: $DATA_PATH"
echo ""

python ./src/main.py \
    --exp $EXP \
    --language $LANGUAGE \
    --vision $VISION \
    --expert $EXPERT \
    --lora $LORA \
    --dataset $DATA_PATH \
