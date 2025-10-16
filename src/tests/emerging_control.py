import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.test_utils import *
from sklearn.metrics.pairwise import cosine_similarity
from rag import ContextManager

# for codebook fetching
import requests
from bs4 import BeautifulSoup

# args
parser = argparse.ArgumentParser()
parser.add_argument('--cache_dir', type=str, default='./.cache')
parser.add_argument('--model_path', type=str, default='cemag/cipher_printing')
parser.add_argument('--language', type=bool, default=False)
parser.add_argument('--vision', type=bool, default=False)
parser.add_argument('--expert', type=bool, default=True)
parser.add_argument('--lora', type=bool, default=False)
parser.add_argument('--instructions_path', type=str, default='./prompts/emergent_control.txt')
parser.add_argument('--rag', action='store_true', default=False)
parser.add_argument('--codebook', action='store_true', default=False)
parser.add_argument('--facts_path', type=str, default='src/RAG/processed_facts_openai.json')
parser.add_argument('--context', type=int, default=5)
parser.add_argument('--scenarios_path', type=str, default='./prompts/unknown_scenarios.json')
parser.add_argument('--num_questions', type=int, default=10)
parser.add_argument('--results_path', type=str, default='./results')
args = parser.parse_args()

results_path = f'{args.results_path}/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(results_path, exist_ok=True)

# Initialize ContextManager for RAG
context_manager = ContextManager(cache_dir=args.cache_dir)

## init experiment
experiment={
    'language':args.language,
    'vision':args.vision,
    'expert':args.expert,
    'lora':args.lora,
    'path':args.model_path
}

# Load instructions
with open(args.instructions_path, "r") as file:
    prompt_template = file.read()

# Load case tests from the JSON file
with open(args.scenarios_path, "r") as file:
    case_tests = json.load(file)

def fetch_gcode_details(hyperlink):
    response = requests.get(hyperlink)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        details = soup.select_one('body > div.container.detail > div > div.col-lg-9.col-md-8.main')
        return details.text if details else "No details found"
    else:
        return f"Error fetching details: {response.status_code}"

playbook = {
    "M104 - Set Hotend Temperature": {
    "function": "M104",
    "description": "Set a new target hot end temperature.",
    "hyperlink": "https://marlinfw.org/docs/gcode/M104.html"
    },

    "M106 - Set Fan Speed": {
    "function": "M106",
    "description": "Turn on fan and set speed",
    "hyperlink": "https://marlinfw.org/docs/gcode/M106.html"
    },

    "M220 - Set Feed rate": {
    "function": "M220",
    "description": "Set the feed rate for the extruder.",
    "hyperlink": "https://marlinfw.org/docs/gcode/M220.html"
    },

    "M221 - Set Extrusion Flow Rate": {
    "function": "M221",
    "description": "Set the extrusion flow rate for the extruder.",
    "hyperlink": "https://marlinfw.org/docs/gcode/M221.html"
    }
}

def get_entry_by_gcode(gcode_command):
    for key, entry in playbook.items():
        if entry['function'] == gcode_command:
            return entry["hyperlink"]
    return None 

def format_scenario(scenario):
    scenario_info = f"""
############################################################
    actual_flowrate = {scenario['Given Information']['actual_flowrate']}
    firmware_flowrate = {scenario['Given Information']['firmware_flowrate']}
    material = {scenario['Given Information']['material']}
    current_temperature = {scenario['Given Information']['current_temperature']}
    feed_rate = {scenario['Given Information']['feed_rate']}
#############################################################"""
    prompt = prompt_template.replace("{system}", scenario_info)
    return prompt

scenarios = []
for scenario in case_tests:
    scenario = format_scenario(scenario)
    scenarios.append({"question":scenario})

case_control_dataset = batchify(scenarios, batch_size=1)

# Initialize results list
results = []

# load model
model, processor, vision_expert = load_model_and_processor(
    experiment, eval=True
)
print("Model loaded successfully.")

# Test emerging capabilities 
for batch in tqdm(case_control_dataset, desc="Processing emerging capabilities batches"):

    with open("prompts/emerging_control_sc.txt", "r") as file:
        system_message = file.read()

    for step in [0,1]:
        
        # we only add retrieved context in second step
        if step == 1:

            ### RAG
            if args.rag:
                relevant_facts_string = context_manager.find_relevant_facts(answers[0], num_facts=5)
                if relevant_facts_string:
                    system_message += f"\n\nMore relevant information: {relevant_facts_string}"

            ### Codebook
            if args.codebook:
                print("Adding codebook context to examples")
                try:                
                    gcode_command = answers[0].split('gcode')[1].split('```')[0]
                    gcode_command = gcode_command.strip().split(" ")[0]
                    gcode_info = get_entry_by_gcode(gcode_command)
                    details= fetch_gcode_details(gcode_info)
                    system_message += f"\n\nMore relevant information re. the selected gcode command: {details}"
                except:
                    system_message+=f"\n\nMore relevant information re. the selected gcode command: It is likely that the G-code command is not needed."

        # inference, happens in both steps
        with torch.no_grad():

            # Prepare batch
            batch_collated = val_collate_fn_emerging_control(
                batch, 
                processor,
                system_message=system_message
            ).to(model.device)

            # Generate outputs
            outputs = model.generate(
                **batch_collated,
                max_new_tokens=1024,
            )
            
            decoded_outputs = map(processor.decode, outputs)
            answers = answer_extractor(decoded_outputs)

        if step == 1: # we only extract the final answer in second step
            try:
                print(answers)
                gcode_command = answers[0].split('```')[1].split('```')[0]
                print(gcode_command)
            except:
                gcode_command = None

    for example, answer in zip(batch, answers):
        question = example['question']
        results.append({
            "questions": question,
            "command": gcode_command,
            "answer": answer
        })

    # Write the results to a JSON file 
    with open(f'{results_path}/emerging_control_results.json', "w") as file:
        json.dump(results, file, indent=4)