#!/usr/bin/env python3
"""
CIPHER: Hybrid Reasoning for Perception, Explanation, and Autonomous Action in Manufacturing
Christos Margadji, Sebastian Pattinson
Institute for Manufacturing, University of Cambridge
October 2025
"""

import torch
import wandb
import logging
import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from model import load_model_and_processor
from train import train_model
from peft import LoraConfig
from trl import SFTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="CIPHER")
parser.add_argument("--lora", type=bool, default=False)
parser.add_argument("--language", type=bool, default=False)
parser.add_argument("--vision", type=bool, default=False)
parser.add_argument("--expert", type=bool, default=True)
parser.add_argument("--dataset", type=str, default="cemag/tl-caxton")
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
parser.add_argument("--cache_dir", type=str, default="./.cache")
args = parser.parse_args()

# WandB Configuration
exp=args.exp
print(f"config= language:{args.language}/vision:{args.vision}/expert:{args.expert}/lora:{args.lora}")
WANDB_RUN_NAME = f"{exp}: language:{args.language}/vision:{args.vision}/expert:{args.expert}/lora:{args.lora}"

# no need to change this
if args.lora:
    PEFT_CONFIG = LoraConfig(    # PEFT (Lora) Configuration
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["patch_embedding","q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    args.peft_config = PEFT_CONFIG
else:
    args.peft_config = None

# Training Arguments
TRAINING_ARGS = SFTConfig(
    output_dir=os.path.join(args.cache_dir, exp),
    num_train_epochs=1,                     # Number of training epochs
    per_device_train_batch_size=4,          # Batch size per device during training
    gradient_accumulation_steps=8,          # Number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # Use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # Use fused AdamW optimizer
    logging_steps=10,                       # Log every 5 steps
    save_strategy="epoch",                  # Save checkpoint every epoch
    save_total_limit=1,                     # Only keep the most recent checkpoint
    learning_rate=1e-4,                     # Learning rate, based on QLoRA paper
    bf16=False,                             # Use bfloat16 precision
    tf32=True,                              # Use tf32 precision
    max_grad_norm=0.3,                      # Max gradient norm, based on QLoRA paper
    warmup_ratio=0.1,                       # 10% warmup
    lr_scheduler_type="cosine",             # Correct scheduler type
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Use reentrant checkpointing
    dataset_text_field="",                  # Dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # Important for collator
)
TRAINING_ARGS.remove_unused_columns=False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def check_cuda_availability():
    """
    Check CUDA availability.
    """
    logger.info("Checking CUDA availability...")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logger.info(f"CUDA is available.")
        logger.info(f"Device Count       : {device_count}")
        logger.info(f"Current Device ID  : {current_device}")
        logger.info(f"Current Device Name: {device_name}")
    else:
        logger.warning("CUDA is not available.")


def initialize_wandb():
    """
    Initialize WandB.
    """
    logger.info("Initializing Weights & Biases...")
    wandb.init(project="CIPHER", name=WANDB_RUN_NAME)


def authenticate_huggingface():
    """
    Authenticate HF token.
    """
    from huggingface_hub import login
    try:
        load_dotenv()
        login(
            token=os.environ.get("HF_TOKEN")
        )   
        logger.info("Authenticating with Hugging Face Hub...")
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        raise


def load_datasets(dataset):
    """
    Load the hf dataset.
    """
    logger.info("Loading datasets...")
    try:
        train_dataset = load_dataset(dataset, split="train")
        val_dataset = load_dataset(dataset, split="validation")
        logger.info("Datasets successfully loaded.")
        return train_dataset, val_dataset
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise


def main():
    """
    execute the training workflow.
    """
    try:

        check_cuda_availability()
        initialize_wandb()
        authenticate_huggingface()

        train_dataset, validation_dataset = load_datasets(
            args.dataset
        )

        logger.info("Loading model and processor...")
        model, processor, expert = load_model_and_processor(args)

        logger.info("Starting model training...")

        train_model(
            model = model, 
            processor = processor, 
            expert = expert, 
            train_dataset = train_dataset, 
            eval_dataset = validation_dataset,
            training_args = TRAINING_ARGS,
            args = args
            )

        logger.info("Model training complete.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    main()
