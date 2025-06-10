# from data_utils import *
import random 
import json 
from PIL import Image

prompt_template="ss?"
expert_template = """{{flowrate:{FLOW_RATE_VALUE}}}"""

with open("/home/cm2161/Documents/llama-manufacturing/pr-intern/src/data/general_statements.json", 'r') as file:
    general_statements = json.load(file)["general_statements"]
with open("/home/cm2161/Documents/llama-manufacturing/pr-intern/src/data/qual_over_extrusion.json", 'r') as file:
    qual_over_extrusion = json.load(file)["over_extrusion_statements"]
with open("/home/cm2161/Documents/llama-manufacturing/pr-intern/src/data/qual_under_extrusion.json", 'r') as file:
    qual_under_extrusion = json.load(file)["under_extrusion_statements"]
with open("/home/cm2161/Documents/llama-manufacturing/pr-intern/src/data/qual_good_extrusion.json", 'r') as file:
    qual_good_extrusion = json.load(file)["good_extrusion_statements"]
with open("/home/cm2161/Documents/llama-manufacturing/pr-intern/src/data/quant_templates.json", 'r') as file:
    quant_templates = json.load(file)["flow_rate_statements"]


def format_data(sample, image=True, fr= False, train=True):

    formatted_data = {"messages": [{"role": "user","content": []}]}

    # Add the image token if image == True
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
                "content": synthesize_answer(sample)
            }
        )

    return formatted_data


def synthesize_answer(sample):

    gs= random.choice(general_statements)["statement"]

    quant= random.choice(quant_templates)["statement"]
    quant= quant.format(flow_rate=sample["flow_rate"])

    if float(sample['flow_rate']) <90:
        qual= random.choice(qual_under_extrusion)["statement"]
    elif float(sample['flow_rate']) >110:
        qual= random.choice(qual_over_extrusion)["statement"]
    else:
        qual= random.choice(qual_good_extrusion)["statement"]
        
    final_statement= f"{gs} {quant} {qual}"

    return final_statement


sample= {"flow_rate":"150","full_img_path":"/home/cm2161/Documents/llama-manufacturing/pr-intern/pred_vs_gt_epoch_9600000.png"}
print(format_data(sample=sample, fr=100))