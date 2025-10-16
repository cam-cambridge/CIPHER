import os

import pandas as pd
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

login(
    token=os.environ.get("HF_TOKEN")
)

CACHE_DIR = "./.cache"

import sys
# Add both the src directory and the project root to sys.path
_current_file = os.path.abspath(__file__)
_src_dir = os.path.dirname(os.path.dirname(_current_file))  # src directory
_project_root = os.path.dirname(_src_dir)  # CIPHER directory
sys.path.insert(0, _src_dir)
sys.path.insert(0, _project_root)

os.makedirs(CACHE_DIR, exist_ok=True)

def load_model_and_processor(experiment, eval=False):
    
    model_path = experiment["path"]
    
    if experiment["lora"]:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
    
    processor = AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        cache_dir=CACHE_DIR,
    )
    
    if experiment.get('expert'):
        import vexpert
        print("Loading vision expert")
        vision_expert = vexpert.VisionExpert(load_dir=CACHE_DIR)
    else:
        vision_expert = False

    if eval:
        model.eval()

    return model, processor, vision_expert


def load_test_dataset(file_path, base_dir, test_samples):
    """
    Loads and preprocesses the dataset, and randomly samples a specified number of rows.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        base_dir (str): Base directory to prepend to the image paths.
        test_samples (int): Number of random samples to return from the dataset.

    Returns:
        pd.DataFrame: A randomly sampled subset of the dataset.
    """
    data = pd.read_csv(file_path).drop(columns=['nozzle_tip_x', 'nozzle_tip_y'])     # Load and clean the dataset
    sampled_data = data.sample(n=test_samples).reset_index(drop=True)    # Randomly sample test_samples rows
    data= [row for _, row in sampled_data.iterrows()]
    return data


def load_test_dataset_from_hf(dataset_name, split='test', test_samples=None):
    """
    Loads dataset from Hugging Face Hub and randomly samples a specified number of rows.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub (e.g., 'cemag/tl-caxton').
        split (str): Dataset split to load ('train', 'validation', or 'test'). Default is 'test'.
        test_samples (int, optional): Number of random samples to return. If None, returns entire split.

    Returns:
        list: A list of dataset samples as dictionaries.
    """
    from datasets import load_dataset
    
    print(f"Loading dataset '{dataset_name}' (split: {split}) from Hugging Face Hub...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name, split=split, cache_dir=CACHE_DIR)
    
    # Convert to pandas for easier manipulation
    df = dataset.to_pandas()
    
    # Sample if requested
    if test_samples is not None and test_samples < len(df):
        df = df.sample(n=test_samples).reset_index(drop=True)
        print(f"Sampled {test_samples} examples from {split} split")
    else:
        print(f"Using all {len(df)} examples from {split} split")
    
    # Convert image to PIL format and prepare data
    processed_data = []
    for idx, row in df.iterrows():
        sample = {
            'flow_rate': row['flow_rate'],
            'nozzle_tip_x': row['nozzle_tip_x'],
            'nozzle_tip_y': row['nozzle_tip_y'],
            'image': row['image'],  # Already a PIL Image from HF datasets
        }
        processed_data.append(sample)
    
    return processed_data


def batchify(dataset, batch_size):
    return [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]


def val_collate_fn_emerging_control(examples, processor, system_message=None, RAG=False):
    
    if RAG:
        print("Adding RAG context to examples")
        from rag import ContextManager
        context_manager = ContextManager()
        examples = context_manager.add_context_to_examples(examples, num_facts=context)
        templates = [context_manager.format_data_with_system(example, system_message, RAG=RAG) for example in examples]
    else:
        templates= [format_data_ask(example) for example in examples]
        
    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates]

    batch = processor(text=texts, return_tensors="pt", padding=True)

    return batch

def val_collate_fn(examples, processor, expert=False):

    if expert:
        _, _, expert_batch = expert.validate_step(examples)
        templates= [format_data(example, fr=fr, image=True) for example,fr in zip(examples, expert_batch)]
    else:
        templates= [format_data(example, image=True) for example in examples]

    # Extract images directly from the templates before processing
    image_inputs = []
    for template in templates:
        for content in template["messages"][0]["content"]:
            if content["type"] == "image":
                img = content["image"]
                # Convert dict format back to PIL Image if needed
                if isinstance(img, dict) and 'bytes' in img:
                    from io import BytesIO
                    img = Image.open(BytesIO(img['bytes']))
                elif not isinstance(img, Image.Image):
                    # If it's still not a PIL Image, try to convert it
                    img = Image.open(img) if isinstance(img, str) else img
                image_inputs.append(img)
                break
    
    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates] # puts in template, and <image> token is isolated from img

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    return batch

def val_collate_fn_flickr(examples, processor):

    templates= [format_data_flickr(image=example['image']) for example in examples]

    image_inputs = [process_vision_info(template["messages"])[0] for template in templates]
    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates] # puts in template, and <image> token is isolated from img

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    return batch

def val_collate_fn_squad(examples, processor):
    
    templates= [format_data_squad(example) for example in examples]

    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates] # puts in template, and <image> token is isolated from img

    batch = processor(text=texts, return_tensors="pt", padding=True)

    return batch

def val_collate_fn_ask(examples, processor):
    
    templates= [format_data_ask(example) for example in examples]

    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates] # puts in template, and <image> token is isolated from img

    batch = processor(text=texts, return_tensors="pt", padding=True)

    return batch

prompt_template="What do you see?"
expert_template = """{{flowrate:{FLOW_RATE_VALUE}}}"""

def format_data_flickr(image):
    
    formatted_data = {"messages": [{"role": "user","content": []}]}

    if image:
        formatted_data["messages"][0]["content"].append(
                            {
                                "type": "image","image": image,
                            }
                        )

    formatted_data["messages"][0]["content"].append(
        {
            "type": "text", "text": prompt_template
        }
    )

    return formatted_data

def format_data_squad(example):
    
    formatted_data = {"messages": [{"role": "user","content": []}]}

    formatted_data["messages"][0]["content"].append(
        {
            "type": "text", "text": example['context']+" Question: "+example['question']
        }
    )

    return formatted_data

def format_data_ask(example):
    
    formatted_data = {"messages": [{"role": "user","content": []}]}

    formatted_data["messages"][0]["content"].append(
        {
            "type": "text", "text": example
        }
    )

    return formatted_data

def format_data(sample, image=True, fr= False):
    
    formatted_data = {"messages": [{"role": "user","content": []}]}

    if image:
        # Handle both HF dataset (with 'image' field) and local files (with 'full_img_path')
        if 'image' in sample and sample['image'] is not None:
            img = sample['image']  # Already a PIL Image from HF
        elif 'full_img_path' in sample and sample['full_img_path'] is not None:
            img = Image.open(sample["full_img_path"])  # Open from local path
        else:
            raise ValueError("Sample must have either 'image' or 'full_img_path' field")
        
        formatted_data["messages"][0]["content"].append(
            {
                "type": "image", "image": img,
            }
        )

    if fr:
        formatted_data["messages"][0]["content"].append(
                {
                    "type": "text", "text": expert_template.format(FLOW_RATE_VALUE=fr)+prompt_template
                }
            )
    else:
        formatted_data["messages"][0]["content"].append(
            {
                "type": "text", "text": prompt_template
            }
        )

    return formatted_data

def answer_extractor(examples, start_marker="<|start_header_id|>assistant<|end_header_id|>", end_marker="<|eot_id|>"):
    """
    Extracts content between start_marker and end_marker in the input strings.
    
    Args:
        examples (str or list of str): A single string or list of strings to process.
        start_marker (str): The starting marker to look for
        end_marker (str): The ending marker to look for
        
    Returns:
        list of str: Extracted content from each input string.
    """

    if isinstance(examples, str):
        examples = [examples]  # Convert single string to list for uniform processing
    
    extracted_content = []
    for example in examples:
        # Escape special regex characters in markers
        escaped_start = re.escape(start_marker)
        escaped_end = re.escape(end_marker)
        pattern = f"{escaped_start}\\n\\n(.*?)({escaped_end}.*)?$"
        match = re.search(pattern, example, re.DOTALL)
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
                values.append(None)
        else:
            flowrate_value = None       
            values.append(None)
            
    return values

def format_data_RAG(example, RAG=False):
    
    formatted_data = {"messages": [{"role": "user", "content": []}]}

    formatted_data["messages"][0]["content"].append(
        {
            "type": "text", "text": example["question"]
        }
    )

    if RAG :
        formatted_data["messages"][0]["content"].append(
            {
                "type": "text", "text": example["context"]
            }
        )

    return formatted_data


def batchify_overfit(dataset, batch_size=8):
    batch = []
    for sample in dataset:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Yield remaining samples if any
        yield batch

def val_collate_fn_RAG(examples, processor, system_message=None, RAG=False, context=5):
    
    if RAG:
        print("Adding RAG context to examples")
        from rag import ContextManager
        context_manager = ContextManager()
        examples = context_manager.add_context_to_examples(examples, num_facts=context)
        templates = [context_manager.format_data_with_system(example, system_message, RAG=RAG) for example in examples]
    else:
        templates= [format_data_ask(example) for example in examples]
    
    print(templates[0])

    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates]
    batch = processor(text=texts, return_tensors="pt", padding=True)

    return batch

def generate_RAG_embedding(text):
    """
    Generate embedding using OpenAI's API.
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"  # Use OpenAI's embedding model
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding for text: {text}\nError: {e}")
        return None
