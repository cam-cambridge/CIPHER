import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pandas as pd
from utils.test_utils import *

# args
parser = argparse.ArgumentParser()
parser.add_argument('--cache_dir', type=str, default='./.cache')
parser.add_argument('--model_path', type=str, default='cemag/cipher_printing')
parser.add_argument('--language', type=bool, default=False)
parser.add_argument('--vision', type=bool, default=False)
parser.add_argument('--expert', type=bool, default=True)
parser.add_argument('--lora', type=bool, default=False)
parser.add_argument('--template_path', type=str, default='./prompts/geometry_expert.txt')
parser.add_argument('--geometry_request', type=str, default='a 20mm dimaeter circle.')
parser.add_argument('--results_path', type=str, default='./results')
parser.add_argument('--check_complexity', type=bool, default=False)
args = parser.parse_args()

results_path = f'{args.results_path}/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(results_path, exist_ok=True)

## init experiment
experiment={
    'language':args.language,
    'vision':args.vision,
    'expert':args.expert,
    'lora':args.lora,
    'path': args.model_path
}

# load question
with open(args.template_path, "r") as file:
    template = file.read()

# Avoid str.format on the whole template because it contains braces used in examples
# which would be treated as format fields (e.g., {x_start:.3f}). We only want to
# substitute the explicit placeholder token "geometry_request" in the prompt.
question = template.replace('geometry_request', args.geometry_request)

# load model
model, processor, vision_expert = load_model_and_processor(
    experiment, eval=True
)
print("Model loaded successfully.")

with torch.no_grad():

    # Complexity test
    complexity = True
    if args.check_complexity:
        complexity = complexity_analysis(question)
        print(f"Complexity: {complexity}")

    if complexity:
        # Prepare sample
        batch_collated = val_collate_fn_ask([question], processor).to(model.device)

        # Generate outputs
        outputs = model.generate(
            **batch_collated,
            max_new_tokens=4096,
            temperature=0.2
        )

        # Process results
        decoded_output = processor.decode(outputs[0])
        answer = answer_extractor([decoded_output])[0]

        # Store results
        ask_answers = [answer]
        
        # Save QA results
        results_df = pd.DataFrame({
            'answers': [answer],
        })
        results_df.to_csv(f'{results_path}/geometry_expert_results.csv',
                        index=False)
    else:
        print("Complexity too high, forward to generator...")
        print("This will need an installation of shap-e, please refer to the README.md file for instructions. Or follow this tutorial https://lablab.ai/t/shape-e-tutorial-how-to-set-up-and-use-shap-e-model")

print("Answer saved to: ", results_path)