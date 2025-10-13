import pandas as pd
from PIL import Image
import json
import random
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download
from datasets import load_dataset as hf_load_dataset

prompt_template="What do you see?"
expert_template = """{{flowrate:{FLOW_RATE_VALUE}}}"""
answer_template="""The flowrate is {FLOW_RATE_VALUE}."""

# Load JSON templates from HuggingFace
def _load_json_from_hf(filename, cache_dir="./.cache"):
    """Load JSON file from HuggingFace repository."""
    file_path = hf_hub_download(
        repo_id="cemag/tl-caxton",
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir
    )
    with open(file_path, 'r') as file:
        return json.load(file)

general_statements = _load_json_from_hf("general_statements.json")["general_statements"]
qual_over_extrusion = _load_json_from_hf("qual_over_extrusion.json")["over_extrusion_statements"]
qual_under_extrusion = _load_json_from_hf("qual_under_extrusion.json")["under_extrusion_statements"]
qual_good_extrusion = _load_json_from_hf("qual_good_extrusion.json")["good_extrusion_statements"]
quant_templates = _load_json_from_hf("quant_templates.json")["flow_rate_statements"]

## note: reinstall bitsandbytes if not working. enforced version 0.37.2.

def synthesize_answer(sample, general=True, quant=True, qual=True):
    final_statement = ""

    # General statement
    if general:
        gs = random.choice(general_statements)["statement"]
        final_statement += f"{gs} "

    # Quantitative statement
    if quant:
        quant_statement = random.choice(quant_templates)["statement"]
        quant_statement = quant_statement.format(flow_rate=sample["flow_rate"])
        final_statement += f"{quant_statement} "

    # Qualitative statement
    if qual:
        flow_rate = float(sample['flow_rate'])
        if flow_rate < 90:
            qual_statement = random.choice(qual_under_extrusion)["statement"]
        elif flow_rate > 110:
            qual_statement = random.choice(qual_over_extrusion)["statement"]
        else:
            qual_statement = random.choice(qual_good_extrusion)["statement"]
        final_statement += f"{qual_statement}"

    return final_statement.strip()


def format_data(sample, image=True, fr= False, train=True):

    formatted_data = {"messages": [{"role": "user","content": []}]}

    if image:
        formatted_data["messages"][0]["content"].append(
                            {
                                "type": "image","image": sample["image"],
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
        
    if train:
        formatted_data["messages"].append(
            {
                "role": "assistant",
                "content": synthesize_answer(sample=sample)
            }
        )

    return formatted_data


def collate_vqa(examples, processor, expert):

    if expert:
        _,_,expert_batch = expert.active_learning(examples)
        templates = [format_data(example, fr=fr, image=True, train=True) for example,fr in zip(examples,expert_batch)]
    else:
        templates = [format_data(example, image=True, train=True) for example in examples]

    image_inputs = [process_vision_info(template["messages"])[0] for template in templates]
    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates] # puts in template, and <image> token is isolated from img
    
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels

    return batch


def val_collate_fn(examples, processor, expert=False):

    if expert:
        _, _, expert_batch = expert.validate_step(examples)
        templates= [format_data(example, fr=fr, image=True, train=False) for example,fr in zip(examples, expert_batch)]
    else:
        templates= [format_data(example, image=True, train=False) for example in examples]

    image_inputs = [process_vision_info(template["messages"])[0] for template in templates]
    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates] # puts in template, and <image> token is isolated from img
    
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    return batch


def load_dataset(split="train", n_rows=None, cache_dir="./.cache"):
    """
    Load dataset from HuggingFace.
    
    Args:
        split (str): Which split to load ('train', 'validation', or 'test')
        n_rows (int, optional): Number of rows to load. If None, loads all rows.
        cache_dir (str): Directory to cache the dataset
        
    Returns:
        list: List of dictionaries containing the dataset samples
    """
    # Load dataset from HuggingFace
    dataset = hf_load_dataset(
        dataset_name="cemag/tl-caxton", 
        split=split, 
        cache_dir=cache_dir
    )

    # Limit number of rows if specified
    if n_rows is not None:
        dataset = dataset.select(range(min(n_rows, len(dataset))))
    
    # Convert to list of dictionaries
    data = [sample for sample in dataset]
    
    return data

