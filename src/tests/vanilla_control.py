import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from utils.test_utils import *

# args
parser = argparse.ArgumentParser()
parser.add_argument('--cache_dir', type=str, default='./.cache')
parser.add_argument('--model_path', type=str, default='cemag/cipher_printing')
parser.add_argument('--language', type=bool, default=False)
parser.add_argument('--vision', type=bool, default=False)
parser.add_argument('--expert', type=bool, default=True)
parser.add_argument('--lora', type=bool, default=False)
parser.add_argument('--prompt_path', type=str, default='./prompts/vanilla_control.txt')
parser.add_argument('--num_questions', type=int, default=10)
parser.add_argument('--results_path', type=str, default='./results')
args = parser.parse_args()

results_path = f'{args.results_path}/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(results_path, exist_ok=True)

## init experiment
experiment={
    'language':args.language,
    'vision':args.vision,
    'expert':args.expert,
    'lora':args.lora,
    'path':args.model_path
}

# Create control questions
def generate_firmware(estimate, error_std=10, bias=0):
    noise = np.random.normal(loc=0, scale=error_std)
    firmware = estimate + bias + noise
    firmware = max(min(firmware, estimate + 30), estimate - 30)
    return firmware
    

print("Creating scenarios...")

with open(args.prompt_path, "r") as file:
    prompt_template = file.read()

tests = []
for _ in range(args.num_questions):
    E = random.randint(30, 300) 
    F = generate_firmware(E)
    ANSWER = (100 / E) * F

    tests.append({"context":prompt_template.format(E=E, F=F), "answer":ANSWER})

naive_control_dataset = batchify(tests, batch_size=8)

print("Scenarios generated successfully.")

# Initialize experiment tracking variables
experiment_answers = []
experiment_answers_original = []

# load model
model, processor, vision_expert = load_model_and_processor(
    experiment, eval=True
)
print("Model loaded successfully.")

for batch in tqdm(naive_control_dataset, desc="Processing domain control batches"):
    with torch.no_grad():

        # Prepare batch
        batch_collated = val_collate_fn_ask(
            batch, 
            processor
        ).to(model.device)

        # Generate outputs
        outputs = model.generate(
            **batch_collated,
            max_new_tokens=1024,
            temperature=0.2
        )

        # Process results
        decoded_outputs = map(processor.decode, outputs)
        answers = answer_extractor(decoded_outputs)

        answers_processed = []
        for answer in answers:
            try:
                answers_processed.append(round(float(answer.split('M221 S')[1].split("'")[0]), 2))
            except:
                try:
                    answers_processed.append(round(float(answer.split('M221 S')[1].split("`")[0]), 2))
                except:                 
                    answers_processed.append(100.0)
                    print(answer)

        answers = answers_processed

        original_answers = [round(float(item['answer']), 2) for item in batch]
        experiment_answers.extend(answers)
        experiment_answers_original.extend(original_answers)

        # Save vision results
        results_df = pd.DataFrame({
            'answers': experiment_answers,
            'answers_original': experiment_answers_original,
        })
        results_df.to_csv(f'{results_path}/vanilla_control_results.csv',
                        index=True,
                        index_label='control_id')
