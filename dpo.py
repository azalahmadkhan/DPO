from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
# import bitsandbytes as bnb
import torch
from datasets import Dataset
import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('preference_data.csv')

# Initialize lists
prompts = []
selected = []
rejected = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    prompt = row['prompt']
    accept = row['accepted']
    reject = row['rejected']

    prompts.append(prompt)
    selected.append(accept)
    rejected.append(reject)

# prompts = prompts[:5]
# selected = selected[:5]
# rejected = rejected[:5]
# Create a dictionary in the desired format
dpo_dataset_dict = {
    "prompt": prompts,
    "chosen": selected,
    "rejected": rejected,
}
del df
dataset = Dataset.from_dict(dpo_dataset_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")           # Load the tokenizer
tokenizer.pad_token = tokenizer.eos_token  # Explicitly set padding token
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure pad_token_id is set

model = AutoModelForCausalLM.from_pretrained(               # Load the pre-trained model weights
    "gpt2",
    state_dict=torch.load("sft_gpt.bin"),
).to(device)

training_args = DPOConfig(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    num_train_epochs=3,
    save_steps= 100,
    learning_rate=2e-4,
    # bf16=False,
    save_total_limit=3,
    logging_steps=10,
    output_dir="DPOP",
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    remove_unused_columns=False,
    beta=0.1,
    max_prompt_length=128,  # Move these here
    max_length=256,
    padding_value=tokenizer.pad_token_id
    # output_dir="./dpo_output" 
)


dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    # config=DPOConfig(
    #         beta=0.1,
    #         max_prompt_length=128,  # Move these here
    #         max_length=256,
    #         output_dir="./dpo_output" 
    #     )
)

dpo_trainer.train()

dpo_trainer.save_model("gpt_5")
