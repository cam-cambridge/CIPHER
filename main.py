import torch
import wandb
import logging

from data.data_utils import load_dataset
from src.model import load_model_and_processor
from src.train import train_model
from src.config import (
    WANDB_PROJECT,
    WANDB_RUN_NAME,
    HF_TOKEN,
    TRAIN_PATH,
    VAL_PATH,
    BASE_DIR,
    N_TRAIN_ROWS,
    N_VAL_ROWS,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def check_cuda_availability():
    """
    Check and log CUDA availability and device details.
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
    Initialize Weights & Biases for experiment tracking.
    """
    logger.info("Initializing Weights & Biases...")
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)


def authenticate_huggingface():
    """
    Authenticate with the Hugging Face Hub.
    """
    from huggingface_hub import login
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    logger.info("Authenticating with Hugging Face Hub...")
    login(token=HF_TOKEN)


def load_datasets(train_path, val_path, base_dir, n_train_rows, n_val_rows):
    """
    Load training and validation datasets.
    """
    logger.info("Loading datasets...")
    try:
        train_dataset = load_dataset(train_path, base_dir, n_rows=n_train_rows)
        val_dataset = load_dataset(val_path, base_dir, n_rows=n_val_rows)
        logger.info("Datasets successfully loaded.")
        return train_dataset, val_dataset
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise


def main():
    """
    Main function to execute the training workflow.
    """
    try:

        check_cuda_availability()
        initialize_wandb()
        authenticate_huggingface()

        train_dataset, validation_dataset = load_datasets(
            TRAIN_PATH, VAL_PATH, BASE_DIR, N_TRAIN_ROWS, N_VAL_ROWS
        )

        logger.info("Loading model and processor...")
        model, processor, expert = load_model_and_processor()

        logger.info("Starting model training...")

        train_model(
            model = model, 
            processor = processor, 
            expert = expert, 
            train_dataset = train_dataset, 
            eval_dataset = validation_dataset
            )

        logger.info("Model training complete.")

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise


if __name__ == "__main__":
    main()
