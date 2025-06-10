import re
import os
import ast
from torchvision import transforms
from PIL import Image
import uuid
from matplotlib import pyplot as plt 
from tabulate import tabulate

def load_and_augment_images_torchvision(image_paths):
    """
    Load images from file paths, apply all augmentations using torchvision,
    and convert the resulting tensors back to PIL images.
    
    Parameters:
        image_paths (list of str): List of file paths to images.
    
    Returns:
        list of PIL.Image.Image: List of augmented images.
    """
    # Define the augmentation pipeline
    transform_pipeline = transforms.Compose([
        transforms.RandomAffine(degrees=20,                        # Random rotation (±30 degrees)
                                translate=(0.1, 0.1)),             # Random translation (10% max)
        transforms.RandomResizedCrop(size=200, scale=(0.8, 1.2)),  # Random scaling and cropping
        transforms.RandomHorizontalFlip(p=0.5),                    # Random horizontal flip
        transforms.ColorJitter(brightness=0.5,                     # Random brightness adjustment
                               contrast=0.5,                       # Random contrast adjustment
                               hue=0.5),                           # Random hue adjustment
        transforms.ToTensor(),                                     # Convert PIL image to PyTorch tensor
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],                # Normalize tensor values
        #                      std=[0.5, 0.5, 0.5]),
        transforms.ToPILImage()                                    # Convert tensor back to PIL image
    ])

    augmented_images = []
    
    for path in image_paths:
        img = Image.open(path["full_img_path"])
        img = transform_pipeline(img)  # Apply augmentations
        augmented_images.append(img)   # Add to results list
        # random_name = f"{uuid.uuid4()}.jpg"  # Generate a random name
        # save_path = os.path.join("/home/cm2161/Documents/llama-manufacturing/load_test", random_name)
        # img.save(save_path)  # Save the augmented image

    return augmented_images

def answer_extractor(examples, start_marker="<|start_header_id|>assistant<|end_header_id|>", end_marker="<|eot_id|>"):
    """
    Extracts content between <|end_header_id|> and <|eot_id|> in the input strings.
    
    Args:
        examples (str or list of str): A single string or list of strings to process.
        
    Returns:
        list of str: Extracted content from each input string.
    """

    if isinstance(examples, str):
        examples = [examples]  # Convert single string t do list for uniform processing
    
    extracted_content = []
    for example in examples:
        match = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(<\|eot_id\|>.*)?$", example, re.DOTALL)
        extracted_content.append(match.group(1) if match else None)

    return extracted_content    

def question_extractor(examples, start_marker="<|start_header_id|>user<|end_header_id|>", end_marker="<|eot_id|>"):
    """
    Extracts content between <|end_header_id|> and <|eot_id|> in the input strings.
    
    Args:
        examples (str or list of str): A single string or list of strings to process.
        
    Returns:
        list of str: Extracted content from each input string.
    """

    if isinstance(examples, str):
        examples = [examples]  # Convert single string t do list for uniform processing
    
    extracted_content = []
    for example in examples:
        match = re.search(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)(<\|eot_id\|>.*)?$", example, re.DOTALL)
        extracted_content.append(match.group(1) if match else None)

    return extracted_content   

def extract_fr(examples):
    # Ensure `examples` is a list for uniform processing
    
    if isinstance(examples, str):
        examples = [examples]

    values=[]
    for example in examples:

        if example is not None and isinstance(example, str):
            numeric_value = re.findall(r'\b\d+(?:\.\d+)?\b', example)
            if numeric_value:
                flowrate_value = float(numeric_value[0])
                values.append(flowrate_value)
            else:
                flowrate_value = None       
                values.append(100)
        else:
            flowrate_value = None       
            values.append(100)
            
    return values


import uuid

def save_image_tensor(tensor, save_dir, file_name=None):
    """
    Save a 3D tensor (C, H, W) as an image file with a specified or random name.
    
    Parameters:
        tensor (torch.Tensor): A 3D tensor with shape (C, H, W).
        save_dir (str): Directory where the image will be saved.
        file_name (str, optional): Name of the saved file. If None, a random name is generated.
    
    Returns:
        str: The full path of the saved image.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Check tensor dimensions
    if tensor.ndim != 3:
        raise ValueError("Expected a tensor with 3 dimensions (C, H, W).")
    
    # Convert to PIL image
    pil_image = transforms.functional.to_pil_image(tensor)
    
    # Use the given name or generate a random one
    if file_name is None:
        file_name = f"{uuid.uuid4()}.png"  # Generate random name if none provided
    elif not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise ValueError("File name must end with a valid image extension (.png, .jpg, .jpeg).")
    
    save_path = os.path.join(save_dir, file_name)
    
    # Save the image
    pil_image.save(save_path)
    print(f"Saved image to: {save_path}")
    return save_path

def plot_images_with_answers(self, entries, answers):
        """
        Plots images in one column with their corresponding answers displayed next to them.
        
        Args:
        - entries: List of entries containing image paths.
        - answers: List of answers corresponding to the images.
        """
        num_entries = len(entries)
        fig, axs = plt.subplots(num_entries, 2, figsize=(10, num_entries * 2))
        
        if num_entries == 1:  # If only one image, adjust axes for a single subplot
            axs = [axs]

        for idx, (entry, answer) in enumerate(zip(entries, answers)):
            # Load and display the image
            image = Image.open(entry["full_img_path"])
            axs[idx][0].imshow(image)
            axs[idx][0].axis('off')  # Hide axis for the image
            
            # Write the answer
            axs[idx][1].text(0.5, 0.5, answer, fontsize=12, ha='center', va='center', wrap=True)
            axs[idx][1].axis('off')  # Hide axis for the text

        plt.tight_layout()
        plt.savefig("test.jpg")

def chunkify(lst, n):
    """Split a list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# print("Batch structure:")
# for key, value in batch.items():
#     print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")


def config_summary(L_TRAINABLE, V_TRAINABLE, EXPERT, LORA, CACHE_DIR, 
                            MODEL_NAME, TRAIN_PATH, VAL_PATH, BASE_DIR, HF_TOKEN, 
                            N_TRAIN_ROWS, N_VAL_ROWS, TRAINING_ARGS, PEFT_CONFIG=None):
    """
    Generate a formatted summary of configuration settings.

    Parameters:
    - L_TRAINABLE (bool): Whether the language module is trainable.
    - V_TRAINABLE (bool): Whether the vision module is trainable.
    - EXPERT (bool): Whether expert mode is enabled.
    - LORA (bool): Whether LoRA is enabled.
    - CACHE_DIR (str): Directory for caching models.
    - MODEL_NAME (str): Name of the model.
    - TRAIN_PATH (str): Path to the training dataset.
    - VAL_PATH (str): Path to the validation dataset.
    - BASE_DIR (str): Base directory for data.
    - HF_TOKEN (str): Hugging Face token.
    - N_TRAIN_ROWS (int): Number of rows in the training dataset.
    - N_VAL_ROWS (int): Number of rows in the validation dataset.
    - TRAINING_ARGS (SFTConfig): Training arguments.
    - PEFT_CONFIG (LoraConfig, optional): LoRA configuration (if applicable).

    Returns:
    - str: A formatted string summary of the configuration.
    """
    # Add LoRA Configuration conditionally
    if LORA and PEFT_CONFIG:
        lora_config_section = {
            "LoRA Configuration": {
                "Alpha": PEFT_CONFIG.lora_alpha,
                "Dropout": PEFT_CONFIG.lora_dropout,
                "Rank (r)": PEFT_CONFIG.r,
                "Target Modules": PEFT_CONFIG.target_modules,
            }
        }
    else:
        lora_config_section = {}

    # Configuration dictionary
    config_summary = {
        "Trainable Modules": {
            "Language Trainable": L_TRAINABLE,
            "Vision Trainable": V_TRAINABLE,
            "Expert Mode": EXPERT,
            "LoRA Enabled": LORA,
        },
        "Paths & Tokens": {
            "Cache Directory": CACHE_DIR,
            "Model Name": MODEL_NAME,
            "Training Data Path": TRAIN_PATH,
            "Validation Data Path": VAL_PATH,
            "Base Data Directory": BASE_DIR,
            "HF Token": HF_TOKEN[:6] + "..." + HF_TOKEN[-4:],  # Partially hide token
        },
        "Data Parameters": {
            "Training Rows": N_TRAIN_ROWS,
            "Validation Rows": N_VAL_ROWS,
        },
        **lora_config_section,  # Add LoRA Configuration dynamically
        "Training Arguments": {
            "Output Directory": TRAINING_ARGS.output_dir,
            "Epochs": TRAINING_ARGS.num_train_epochs,
            "Batch Size (Train)": TRAINING_ARGS.per_device_train_batch_size,
            "Gradient Accumulation Steps": TRAINING_ARGS.gradient_accumulation_steps,
            "Gradient Checkpointing": TRAINING_ARGS.gradient_checkpointing,
            "Optimizer": TRAINING_ARGS.optim,
            "Logging Steps": TRAINING_ARGS.logging_steps,
            "Save Strategy": TRAINING_ARGS.save_strategy,
            "Learning Rate": TRAINING_ARGS.learning_rate,
            "bfloat16": TRAINING_ARGS.bf16,
            "tf32": TRAINING_ARGS.tf32,
            "Max Gradient Norm": TRAINING_ARGS.max_grad_norm,
            "Warmup Ratio": TRAINING_ARGS.warmup_ratio,
            "LR Scheduler": TRAINING_ARGS.lr_scheduler_type,
        },
    }

    # Create formatted summary
    formatted_summary = ""
    for section, parameters in config_summary.items():
        formatted_summary += f"\n\n{section}:\n"
        table = [[key, value] for key, value in parameters.items()]
        formatted_summary += tabulate(table, headers=["Parameter", "Value"], tablefmt="grid")

    print(formatted_summary)

