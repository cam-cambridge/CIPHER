import pandas as pd
from PIL import Image
import json
import random
from qwen_vl_utils import process_vision_info

prompt_template="What do you see?"
expert_template = """{{flowrate:{FLOW_RATE_VALUE}}}"""
answer_template="""The flowrate is {FLOW_RATE_VALUE}."""

with open("/home/cm2161/rds/hpc-work/llama-manufacturing/pr-intern/src/data/general_statements.json", 'r') as file:
    general_statements = json.load(file)["general_statements"]
with open("/home/cm2161/rds/hpc-work/llama-manufacturing/pr-intern/src/data/qual_over_extrusion.json", 'r') as file:
    qual_over_extrusion = json.load(file)["over_extrusion_statements"]
with open("/home/cm2161/rds/hpc-work/llama-manufacturing/pr-intern/src/data/qual_under_extrusion.json", 'r') as file:
    qual_under_extrusion = json.load(file)["under_extrusion_statements"]
with open("/home/cm2161/rds/hpc-work/llama-manufacturing/pr-intern/src/data/qual_good_extrusion.json", 'r') as file:
    qual_good_extrusion = json.load(file)["good_extrusion_statements"]
with open("/home/cm2161/rds/hpc-work/llama-manufacturing/pr-intern/src/data/quant_templates.json", 'r') as file:
    quant_templates = json.load(file)["flow_rate_statements"]

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


def load_dataset(file_path, base_dir, n_rows=100):
    data = pd.read_csv(file_path)
    data= data.drop(
        columns=["Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0", "timestamp", "nozzle_tip_x", "nozzle_tip_y"]
    ).head(n_rows)
    data["full_img_path"] = base_dir + data["img_path"].astype(str)
    data= [row for _, row in data.iterrows()]
    return data

