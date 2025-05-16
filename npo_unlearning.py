import torch
from torch import nn
from transformers import Trainer, TrainingArguments
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

torch.manual_seed(0)

# snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning', token=hf_token, local_dir='semeval25-unlearning-model')
snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning', token=hf_token, local_dir='semeval25-unlearning-1B-model')

## Fetch and load dataset:
snapshot_download(repo_id='llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public', token=hf_token, local_dir='semeval25-unlearning-data', repo_type="dataset")

# Load tokenizer and model
model = AutoModelForCausalLM.from_pretrained("semeval25-unlearning-1B-model")
# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")

# Custom negative preference loss
class NegativePreferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits_chosen, logits_rejected):
        logp_chosen = torch.log_softmax(logits_chosen, dim=-1)
        logp_rejected = torch.log_softmax(logits_rejected, dim=-1)
        loss = -(logp_chosen.mean() - logp_rejected.mean())
        return loss

# Dataset for preference training
class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.pairs = data
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        entry = self.pairs[idx]

        chosen = self.tokenizer(entry["prompt"] + entry["chosen"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        rejected = self.tokenizer(entry["prompt"] + entry["rejected"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids_chosen": chosen["input_ids"].squeeze(),
            "attention_mask_chosen": chosen["attention_mask"].squeeze(),
            "input_ids_rejected": rejected["input_ids"].squeeze(),
            "attention_mask_rejected": rejected["attention_mask"].squeeze(),
        }

# Custom trainer with preference loss
class PreferenceTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = NegativePreferenceLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])
        outputs_rejected = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"])

        loss = self.loss_fn(outputs_chosen.logits, outputs_rejected.logits)
        return (loss, outputs_chosen) if return_outputs else loss

# Sample preference data
preference_data = [
    {
        "prompt": "Input: How do I reset my password?\nAnswer:",
        "chosen": " You can reset it by clicking 'Forgot Password' on the login screen.",
        "rejected": " Passwords are not important.",
    },
    # Add more examples as needed
]

# Initialize dataset and training args
train_dataset = PreferenceDataset(tokenizer, preference_data)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    learning_rate=5e-5,
)

# Start fine-tuning with custom loss
trainer = PreferenceTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
