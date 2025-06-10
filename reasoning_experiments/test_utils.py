import pandas as pd
from transformers import AutoModelForVision2Seq, AutoProcessor
import re
import openai
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CACHE_DIR = "/home/cm2161/rds/hpc-work/"

def load_model_and_processor(experiment):

    if experiment["lora"]:
        model = AutoModelForVision2Seq.from_pretrained(
            experiment['path'],
            device_map="auto",
            cache_dir=CACHE_DIR
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            experiment['path'],
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
    
    processor = AutoProcessor.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        cache_dir=CACHE_DIR,
    )
    
    if experiment.get('expert'):
        from src.vexpert import VisionExpert
        print("Loading vision expert")
        vision_expert = VisionExpert(load_dir=CACHE_DIR)
    else:
        vision_expert = False

    return model, processor, vision_expert


def load_test_dataset(file_path="/home/cm2161/rds/hpc-work/tl-caxton/cleaned_test_global2.csv", 
                      base_dir="/home/cm2161/rds/hpc-work/tl-caxton/cropped_data/", 
                      test_samples=1000):
    """
    Loads and preprocesses the dataset, and randomly samples a specified number of rows.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        base_dir (str): Base directory to prepend to the image paths.
        test_samples (int): Number of random samples to return from the dataset.

    Returns:
        pd.DataFrame: A randomly sampled subset of the dataset.
    """
    data = pd.read_csv(file_path).drop(columns=['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'timestamp', 'nozzle_tip_x', 'nozzle_tip_y'])     # Load and clean the dataset
    data['full_img_path'] = base_dir + data['img_path'].astype(str) # Add full image path
    sampled_data = data.sample(n=test_samples).reset_index(drop=True)    # Randomly sample test_samples rows
    data= [row for _, row in sampled_data.iterrows()]
    return data


def batchify(dataset, batch_size):
    return [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]


def val_collate_fn_text(examples, processor):
    
    templates= [format_data(example['context']) for example in examples]

    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates]

    batch = processor(text=texts, return_tensors="pt", padding=True)

    return batch


def format_data(example, RAG=False):
    
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

def val_collate_fn(examples, processor, system_message=None, RAG=False):
    
    if RAG:
        for example in examples:
            embedding = generate_RAG_embedding(example["question"])
            if embedding is not None:
                similarities = []
                for fact in processed_facts:
                    embedding_array = np.array(embedding)  # Convert to NumPy array
                    embedding_reshaped = embedding_array.reshape(1, -1)  # Reshape to 2D
                    similarity = cosine_similarity(embedding_reshaped, np.array(fact['embedding']).reshape(1, -1))
                    similarities.append(similarity)

                # Keep N most relevant facts
                N = 5  # Set the number of relevant facts to keep
                top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:N]
                relevant_facts = [processed_facts[i] for i in top_indices]
                relevant_facts_string = "Here is some context that might be useful: " + " | ".join([fact['original_fact'] for fact in relevant_facts])
                example['context'] = relevant_facts_string  # Store relevant facts in a new subdir in context

    templates = [format_data_with_system(example, system_message, RAG=RAG) for example in examples]
    
    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates]

    batch = processor(text=texts, return_tensors="pt", padding=True)

    return batch

def format_data_with_system(example, system_message=None, RAG=False):

    formatted_data = {"messages": [
        {"role": "system", "content": [{"type": "text", "text": system_message}],},
        {"role": "user", "content": []}
        ]
    }

    formatted_data["messages"][1]["content"].append(
        {
            "type": "text", "text": example["question"]
        }
    )

    if RAG :
        formatted_data["messages"][1]["content"].append(
            {
                "type": "text", "text": example["context"]
            }
        )

    return formatted_data


def set_openai_api_key(api_key):
    """Set the OpenAI API key."""
    openai.api_key = api_key

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


def load_processed_facts(file_path):
    """Load processed facts from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

set_openai_api_key("...")

processed_facts = load_processed_facts("/home/cm2161/Documents/llama-manufacturing/pr-intern/src/RAG/processed_facts_openai.json")