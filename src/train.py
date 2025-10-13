import os
import numpy as np
import wandb
import torch
import random
from sklearn.metrics import mean_absolute_error
from transformers.training_args import TrainingArguments
from transformers import TrainerCallback, TrainerState, TrainerControl
from trl import SFTTrainer
from tqdm import tqdm

from utils.data_utils import collate_vqa, val_collate_fn
from utils.utils import question_extractor, answer_extractor, extract_fr, chunkify

class CustomCallbacks(TrainerCallback):
    def __init__(self, trainer, processor, expert):
        self.trainer = trainer
        self.processor = processor
        self.expert = expert

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return super().on_train_begin(args, state, control, **kwargs)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % state.logging_steps == 0:
            self.trainer.model.eval()
            with torch.no_grad():
                random_entries = random.sample(self.trainer.eval_dataset, min(10, len(self.trainer.eval_dataset)))

                all_y_hat, all_y = [], []
                batch_size = 8

                for batch_entries in tqdm(chunkify(random_entries, batch_size), desc="Validating Batches"):
                    batch = val_collate_fn(batch_entries, self.processor, self.expert if self.expert else None).to(self.trainer.model.device)
                    outputs = self.trainer.model.generate(**batch, max_new_tokens=1024, temperature=0.2)

                    decoded_outputs = map(self.processor.decode, outputs)
                    answers = answer_extractor(decoded_outputs)

                    all_y_hat.extend(extract_fr(answers))
                    all_y.extend(entry["flow_rate"] for entry in batch_entries)

                print("Validation sample")
                first_output = self.processor.decode(outputs[0])
                print("validation, flowrate:     ", random_entries[0]["flow_rate"])
                print("validation, question:     ", question_extractor(first_output))
                print("validation, answer:       ", answer_extractor(first_output))

                mae = mean_absolute_error(np.array(all_y_hat), np.array(all_y))
                print("MAE post LLM: ", mae)
                wandb.log({"MAE": mae}, step=state.global_step)

            self.trainer.model.train()

def train_model(model, processor, expert, train_dataset, eval_dataset, training_args, args):

    print("Starting training...")

    # Configure the trainer with or without PEFT based on LORA flag
    trainer_args = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": lambda examples: collate_vqa(examples, processor, expert),
        "tokenizer": processor.tokenizer,
    }

    if args.lora:
        trainer_args["peft_config"] = args.peft_config

    trainer = SFTTrainer(**trainer_args)

    # Add custom callbacks for additional functionality
    custom_callback = CustomCallbacks(trainer=trainer, processor=processor, expert=expert)
    trainer.add_callback(custom_callback)

    # Start the training process
    trainer.train()
    print("Model trained.")