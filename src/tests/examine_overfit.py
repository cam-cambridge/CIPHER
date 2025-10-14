import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import random
import argparse
from tqdm import tqdm
import torch
import pandas as pd
from utils.test_utils import *

# args
parser = argparse.ArgumentParser()
parser.add_argument('--cache_dir', type=str, default='./.cache')
parser.add_argument('--test_samples', type=int, default=100)
parser.add_argument('--model_path', type=str, default='cemag/cipher_printing')
parser.add_argument('--language', type=bool, default=False)
parser.add_argument('--vision', type=bool, default=False)
parser.add_argument('--expert', type=bool, default=True)
parser.add_argument('--lora', type=bool, default=False)
parser.add_argument('--question_answer_path', type=str, default='squad')
parser.add_argument('--image_caption_path', type=str, default='nlphuji/flickr30k')
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

# loading datasets
flickr30k = load_dataset(args.image_caption_path, streaming=True, split='test').shuffle(seed=42).take(args.test_samples)
squad_data = load_dataset(args.question_answer_path, streaming=True, split='validation').shuffle(seed=42).take(args.test_samples)
print("Datasets loaded successfully.")

# load model
model, processor, vision_expert = load_model_and_processor(
    experiment, eval=True
)
print("Model loaded successfully.")

# TEST 1 : Vision overfitting evaluation 
vision_answers = []
vision_answers_original = []

for sample in tqdm(flickr30k, desc="Processing vision overfitting samples"):
    with torch.no_grad():
        # Prepare sample
        batch_collated = val_collate_fn_flickr([sample], processor).to(model.device)

        # Generate outputs
        outputs = model.generate(
            **batch_collated,
            max_new_tokens=1024,
            temperature=0.2
        )

        # Process results
        decoded_output = processor.decode(outputs[0])
        answer = answer_extractor([decoded_output])[0]
        original_caption = sample['caption']

        # Store results
        vision_answers.append(answer)
        vision_answers_original.append(original_caption)

        # Save vision results
        results_df = pd.DataFrame({
            'answers': vision_answers,
            'captions': vision_answers_original,
        })
        results_df.to_csv(f'{results_path}/flickr_experiment_results.csv',
                        index=True,
                        index_label='image_id')

# TEST 2 : Language overfitting evaluation 
qa_answers = []
qa_answers_original = []

for sample in tqdm(squad_data, desc="Processing language overfitting samples"):

    with torch.no_grad():
        # Prepare sample
        batch_collated = val_collate_fn_squad([sample], processor).to(model.device)

        # Generate outputs
        outputs = model.generate(
            **batch_collated,
            max_new_tokens=1024,
            temperature=0.2
        )

        # Process results
        decoded_output = processor.decode(outputs[0])
        answer = answer_extractor([decoded_output])[0]
        original_answer = sample['answers']['text']

        # Store results
        qa_answers.append(answer)
        qa_answers_original.append(original_answer)

        # Save QA results
        results_df = pd.DataFrame({
            'answers': qa_answers,
            'original_answers': qa_answers_original,
        })
        results_df.to_csv(f'{results_path}/squad_experiment_results.csv',
                        index=True,
                        index_label='question_id')