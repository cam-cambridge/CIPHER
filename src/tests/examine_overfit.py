from datasets import load_dataset
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='cemag/cipher_printing')
args = parser.parse_args()

# Convert flickr30k to batches
def batchify(dataset, batch_size=8):
    batch = []
    for sample in dataset:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Yield remaining samples if any
        yield batch

flickr30k = load_dataset("nlphuji/flickr30k", streaming=True, split='test').shuffle(seed=42).take(100)
squad_data = load_dataset('squad', streaming=True, split='validation').shuffle(seed=42).take(100)

dataloader_flickr = list(batchify(flickr30k, batch_size=8))
dataloader_squad = list(batchify(squad_data, batch_size=8))

# Store results for each experiment
experiment_results = {}

# Load and prepare model components
model, processor, vision_expert = load_model_and_processor(args.model_path)

# Test 2: Vision overfitting evaluation 
vision_answers = []
vision_answers_original = []

for batch in tqdm(dataloader_flickr, desc="Processing vision overfitting batches"):
    with torch.no_grad():
        # Prepare batch
        batch_collated = val_collate_fn_flickr(batch, processor).to(model.device)

        # Generate outputs
        outputs = model.generate(
            **batch_collated,
            max_new_tokens=1024,
            temperature=0.2
        )

        # Process results
        decoded_outputs = map(processor.decode, outputs)
        answers = answer_extractor(decoded_outputs)
        original_captions = [item['caption'] for item in batch]

        # Store results
        vision_answers.extend(answers)
        vision_answers_original.extend(original_captions)

        # Save vision results
        results_df = pd.DataFrame({
            'answers': vision_answers,
            'captions': vision_answers_original,
        })
        results_df.to_csv(f'experiments/flickr_experiment_results_{key}.csv',
                        index=True,
                        index_label='image_id')

# Test 3: Language overfitting evaluation
qa_answers = []
qa_answers_original = []

for batch in tqdm(dataloader_squad, desc="Processing language overfitting batches"):

    with torch.no_grad():
        # Prepare batch
        batch_collated = val_collate_fn_squad(batch, processor).to(model.device)

        # Generate outputs
        outputs = model.generate(
            **batch_collated,
            max_new_tokens=1024,
            temperature=0.2
        )

        # Process results
        decoded_outputs = map(processor.decode, outputs)
        answers = answer_extractor(decoded_outputs)
        original_answers = [item['answers']['text'] for item in batch]

        # Store results
        qa_answers.extend(answers)
        qa_answers_original.extend(original_answers)

        # Save QA results
        results_df = pd.DataFrame({
            'answers': qa_answers,
            'original_answers': qa_answers_original,
        })
        results_df.to_csv(f'experiments/squad_experiment_results_{key}.csv',
                        index=True,
                        index_label='question_id')