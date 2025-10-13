import os 
import torch
import argparse
from tqdm import tqdm
import pandas as pd 
from test_utils import *
from datetime import datetime


## setup args, defaulted to work on cloned version of the repo
parser = argparse.ArgumentParser()
parser.add_argument('--test_samples', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--model_path', type=str, default='cemag/cipher_printing')
parser.add_argument('--data_path', type=str, default='cemag/tl-caxton')
parser.add_argument('--language', type=bool, default=False)
parser.add_argument('--vision', type=bool, default=False)
parser.add_argument('--expert', type=bool, default=True)
parser.add_argument('--lora', type=bool, default=False)

parser.add_argument('--results_path', type=str, default='./results')

args = parser.parse_args()
os.makedirs(args.results_path, exist_ok=True)

## init experiment
experiment={
    'language':args.language,
    'vision':args.vision,
    'expert':args.expert,
    'lora':args.lora,
    'path':args.model_path
}

dataset = load_test_dataset_from_hf(
    dataset_name=args.data_path, 
    split='test',
    test_samples=args.test_samples
    )
dataloader_our_dataset = batchify(dataset, batch_size=args.batch_size)

print(f"Validating: {args.model_path}")

model, processor, vision_expert = load_model_and_processor(
    experiment, eval=True
)

## run
experiment_answers = []
experiment_flow_rates = [] 
numerical_values = []

for batch in tqdm(dataloader_our_dataset, desc="Processing..."):
    with torch.no_grad():

        # Prepare batch
        batch_collated = val_collate_fn(
            batch, 
            processor,
            vision_expert if vision_expert else False
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
        flow_rates = [item["flow_rate"] for item in batch]

        # Store results
        experiment_answers.extend(answers)
        experiment_flow_rates.extend(flow_rates)
        numerical_values = extract_fr(experiment_answers)

        # Save results
        results_df = pd.DataFrame({
            'y': experiment_flow_rates,
            'y_class': [-1 if y < 90 else (1 if y > 110 else 0) for y in experiment_flow_rates],
            'y_hat': extract_fr(experiment_answers),
            'y_class_hat': [-1 if y < 90 else (1 if y > 110 else 0) for y in extract_fr(experiment_answers)],
            'answers': experiment_answers,
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(args.results_path+f"/{timestamp}.csv", 
                        index=True, 
                        index_label='image_id')