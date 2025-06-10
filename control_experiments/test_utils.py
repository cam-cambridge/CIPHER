import pandas as pd
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import re

CACHE_DIR = "/home/cm2161/rds/hpc-work/"

def load_model_and_processor(experiment):
    
    model_path = f"/home/cm2161/rds/hpc-work/{experiment['path']}/checkpoint-279"
    
    if experiment["lora"]:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map="auto",
            cache_dir=CACHE_DIR
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

def val_collate_fn(examples, processor, expert=False):

    if expert:
        _, _, expert_batch = expert.validate_step(examples)
        templates= [format_data(example, fr=fr, image=True) for example,fr in zip(examples, expert_batch)]
    else:
        templates= [format_data(example, image=True) for example in examples]

    image_inputs = [process_vision_info(template["messages"])[0] for template in templates]
    texts = [processor.apply_chat_template(template["messages"], tokenize=False) for template in templates] # puts in template, and <image> token is isolated from img

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # print("Batch structure:")
    # for key, value in batch.items():
    #     print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")
    
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

def format_data(sample, image=True, fr= False):
    
    formatted_data = {"messages": [{"role": "user","content": []}]}

    if image:
        formatted_data["messages"][0]["content"].append(
                            {
                                "type": "image","image": Image.open(sample["full_img_path"]),
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