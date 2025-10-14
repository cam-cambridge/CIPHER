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
parser.add_argument('--question', type=str, default='Ask your question here.')
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

# load model
model, processor, vision_expert = load_model_and_processor(
    experiment, eval=True
)
print("Model loaded successfully.")

with torch.no_grad():
    # Prepare sample
    batch_collated = val_collate_fn_ask([args.question], processor).to(model.device)

    # Generate outputs
    outputs = model.generate(
        **batch_collated,
        max_new_tokens=1024,
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
    results_df.to_csv(f'{results_path}/ask_results.csv',
                    index=True,
                    index_label='question')