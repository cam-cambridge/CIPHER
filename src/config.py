import os
from trl import SFTConfig
from peft import LoraConfig
from src.utils import config_summary

L_TRAINABLE=False
V_TRAINABLE=False
EXPERT=True
LORA=False

# WandB Configuration
WANDB_PROJECT = "CIPHER"
exp="CIPHER_EXPERIMENT_1"
print(f"config= language:{L_TRAINABLE}/vision:{V_TRAINABLE}/expert:{EXPERT}/lora:{LORA}")
WANDB_RUN_NAME = f"{exp}: language:{L_TRAINABLE}/vision:{V_TRAINABLE}/expert:{EXPERT}/lora:{LORA}"

# Hugging Face Login Token
HF_TOKEN = "..."

# Model Configurations
CACHE_DIR = "./cache" # handy if root dir is not
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Data Paths
TRAIN_PATH = "./data/train.csv"
VAL_PATH = "./data/val.csv"
BASE_DIR = "./data/images/"  

# Data Parameters
N_TRAIN_ROWS = 100000
N_VAL_ROWS = 5000

if LORA:
    PEFT_CONFIG = LoraConfig(    # PEFT (Lora) Configuration
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["patch_embedding","q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

# Training Arguments
TRAINING_ARGS = SFTConfig(
    output_dir=os.path.join(CACHE_DIR, exp),
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

# Generate the summary
summary = config_summary(
    L_TRAINABLE=L_TRAINABLE,
    V_TRAINABLE=V_TRAINABLE,
    EXPERT=EXPERT,
    LORA=LORA,
    CACHE_DIR=CACHE_DIR,
    MODEL_NAME=MODEL_NAME,
    TRAIN_PATH=TRAIN_PATH,
    VAL_PATH=VAL_PATH,
    BASE_DIR=BASE_DIR,
    HF_TOKEN=HF_TOKEN,
    N_TRAIN_ROWS=N_TRAIN_ROWS,
    N_VAL_ROWS=N_VAL_ROWS,
    TRAINING_ARGS=TRAINING_ARGS,
    PEFT_CONFIG=PEFT_CONFIG if LORA else None,
)
