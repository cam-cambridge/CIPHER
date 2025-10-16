import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
parser.add_argument('--questions_path', type=str, default='prompts/3d_printing_questions.json')
parser.add_argument('--rag', action='store_true', default=False)
parser.add_argument('--facts_path', type=str, default='src/RAG/processed_facts_openai.json')
parser.add_argument('--context', type=int, default=5)
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

# Load domain expertise questions
with open(args.questions_path, "r") as file:
    domain_questions = json.load(file)["questions"]
domain_qs = []
for question in domain_questions[args.num_questions:]:
    domain_qs.append({"question": question})  # Placeholder for answers
domain_questions_dataset = batchify(domain_qs, batch_size=8)

# load model
model, processor, vision_expert = load_model_and_processor(
    experiment, eval=True
)
print("Model loaded successfully.")

results = []
# Domain-specific QA evaluation
for batch in tqdm(domain_questions_dataset, desc="Processing domain questions batches"):
    with torch.no_grad():
        
        batch_collated = val_collate_fn_RAG(
            batch, 
            processor,
            RAG = args.rag,
            context = args.context,
            facts_path = args.facts_path,
        ).to(model.device)
        
        # Generate outputs
        outputs = model.generate(
            **batch_collated,
            max_new_tokens=1024,
            temperature=0.2
        )
        
        decoded_outputs = map(processor.decode, outputs)
        answers = answer_extractor(decoded_outputs)

        for example, answer in zip(batch, answers):
            question = example['question']
            
            if args.rag:
                context = example['context']
                results.append({
                        "questions": question,
                        "context": context,
                        "answer": answer
                    })
            else:
                results.append({
                    "questions": question,
                    "answer": answer
                })

    # Write the results to a JSON file
    with open(f'{results_path}/domain_expertise_experiment_results.json', "w") as file:
        json.dump(results, file, indent=4)