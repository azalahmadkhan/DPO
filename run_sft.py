# very nice https://huggingface.co/docs/trl/en/sft_trainer

# --- Step 1: Setup & Imports ---
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline # Added for verification step
)
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Step 2: Configuration ---
logger.info("--- Configuring Run ---")

# Specify the Qwen model - using 1.8B as requested
model_name = "Qwen/Qwen1.5-1.8B-Chat"

# *** YOUR DATA FILE PATH ***
# Path to the file containing the paired data with 'source' and 'target' keys
sft_data_file = "train_for_ft.txt" # <--- CHANGE THIS PATH

# *** KEYS in your JSONL data file ***
# Based on your file content example from 'train_for_ft.jsonl'
source_key = "source"
target_key = "target"

# Directory to save the fine-tuned model locally
output_dir = f"./{model_name.split('/')[-1]}-promptist-sft"

# Hyperparameters from Promptist Paper (Appendix A, Table 6, SFT)
paper_learning_rate = 5e-5
paper_max_steps = 15000 # Total training steps
paper_adam_beta1 = 0.9
paper_adam_beta2 = 0.999
paper_adam_epsilon = 1e-6
paper_weight_decay = 0.1
paper_total_batch_size = 256 # Effective batch size target
paper_max_length = 512 # Max sequence length during tokenization

# *** YOUR HARDWARE CONFIGURATION ***
# Adjust per_device_batch_size based on your GPU VRAM. Start low (e.g., 1 or 2).
# For Qwen1.5-1.8B with bf16/fp16, batch size 2 or 4 might work on decent GPUs (e.g., 16GB+ VRAM)
per_device_batch_size = 2
# Calculate gradient accumulation steps
# Assumes single GPU. If multi-GPU, divide paper_total_batch_size by (per_device_batch_size * num_gpus)
gradient_accumulation_steps = paper_total_batch_size // per_device_batch_size
gradient_accumulation_steps = max(1, gradient_accumulation_steps) # Ensure it's at least 1
effective_batch_size = per_device_batch_size * gradient_accumulation_steps

logger.info(f"Using model: {model_name}")
logger.info(f"SFT Data File: {sft_data_file}")
logger.info(f"Output Directory: {output_dir}")
logger.info(f"Target Effective Batch Size: {paper_total_batch_size}")
logger.info(f"Per Device Batch Size: {per_device_batch_size}")
logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
logger.info(f"Calculated Effective Batch Size: {effective_batch_size}")
logger.info(f"Max Sequence Length: {paper_max_length}")
logger.info(f"Max Training Steps: {paper_max_steps}")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Step 3: Load and Prepare Data ---
logger.info("--- Loading and Preparing Data ---")

# Load the dataset
logger.info(f"Loading dataset from {sft_data_file}...")
try:
    # Use 'train' split by default as loaded from single file
    raw_dataset = load_dataset("json", data_files={"train": sft_data_file})['train']
    logger.info("Dataset loaded successfully.")
    logger.info(f"Dataset structure: {raw_dataset}")
    # Check first example and keys
    first_example = raw_dataset[0]
    logger.info(f"First example: {first_example}")
    if source_key not in first_example or target_key not in first_example:
        logger.error(f"Expected keys '{source_key}' and '{target_key}' not found in the first example!")
        exit(1)
except FileNotFoundError:
    logger.error(f"Data file not found at {sft_data_file}. Please check the path.")
    exit(1)
except Exception as e:
    logger.error(f"Error loading or processing dataset: {e}")
    exit(1)

# Load Tokenizer
logger.info(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set pad token if missing (common for Qwen)
if tokenizer.pad_token is None:
    logger.warning("Tokenizer lacks pad token. Setting pad_token = eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
    # Important: Also configure model's pad_token_id later

logger.info(f"Tokenizer pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")

# Updated tokenization function for 'train_for_ft.jsonl' format
# Function using pre-padding within the .map call
def format_and_tokenize(examples):
    full_input_prompts = examples[source_key]
    targets_raw = examples[target_key]

    if not isinstance(full_input_prompts, list): full_input_prompts = [full_input_prompts]
    if not isinstance(targets_raw, list): targets_raw = [targets_raw]

    targets_cleaned = [tgt.replace("<|endoftext|>", "").strip() if isinstance(tgt, str) else "" for tgt in targets_raw]

    valid_indices = [i for i, (inp, tgt) in enumerate(zip(full_input_prompts, targets_cleaned))
                     if isinstance(inp, str) and inp.strip() and isinstance(tgt, str) and tgt.strip()]
    valid_inputs = [full_input_prompts[i] for i in valid_indices]
    valid_targets_cleaned = [targets_cleaned[i] for i in valid_indices]

    if not valid_inputs:
        logger.warning("Entire batch invalid. Returning empty structure.")
        return {"input_ids": [], "attention_mask": [], "labels": []}

    formatted_texts = [f"{inp}{tgt_clean}{tokenizer.eos_token}"
                       for inp, tgt_clean in zip(valid_inputs, valid_targets_cleaned)]

    # --- Tokenization WITH PRE-PADDING ---
    model_inputs = tokenizer(
        formatted_texts,
        max_length=paper_max_length,
        padding="max_length", # Pad all examples to max_length HERE
        truncation=True
        # return_tensors="np" # Optional: return numpy arrays might help sometimes? try without first
    )

    input_part_tokens = tokenizer(
        valid_inputs,
        max_length=paper_max_length,
        truncation=True,
        padding=False # Don't need to pad this length check
    )

    labels = []
    skipped_count = 0
    for i in range(len(model_inputs["input_ids"])):
        try:
            input_len = len(input_part_tokens["input_ids"][i])
        except IndexError:
            logger.warning(f"IndexError getting input length for example index {i}. Using 0.")
            input_len = 0

        current_input_ids = model_inputs["input_ids"][i]
        # IMPORTANT: Create labels as list for modification
        current_labels = list(current_input_ids)

        if input_len >= len(current_labels):
             logger.warning(f"Skipping {i}: Input part ({input_len}) >= total ({len(current_labels)}).")
             skipped_count += 1
             # We MUST ensure labels list has same number of elements as input_ids list eventually
             # Maybe append a dummy list full of -100? Or handle downstream?
             # For now, let's just pad with -100 to match length if skipping
             labels.append([-100] * len(current_input_ids)) # Append dummy label list
             continue

        # Apply masking to non-skipped examples
        for j in range(input_len):
             current_labels[j] = -100

        # Explicitly mask padding tokens in labels *if they weren't part of input*
        # This might be redundant if DataCollator handles it, but let's be sure
        try:
            # Start masking from input_len onwards if it's a pad token
            for k in range(input_len, len(current_labels)):
                 if current_labels[k] == tokenizer.pad_token_id:
                      current_labels[k] = -100
        except Exception as e_pad:
             logger.warning(f"Minor issue during label padding check: {e_pad}")


        labels.append(current_labels)


    # *** Crucial Check after loop ***
    # If we skipped examples, the lengths of the main lists might mismatch the label list
    # This structure MUST return lists of the *same length* for all keys
    # Let's adjust if skipping occurred (this part is tricky)

    # A safer approach might be to NOT skip, but to handle the error cases
    # by creating valid (even if dummy) label lists. Let's revert the skip logic slightly:

    final_labels = []
    for i in range(len(model_inputs["input_ids"])):
        try:
            input_len = len(input_part_tokens["input_ids"][i])
        except IndexError: input_len = 0

        current_input_ids = model_inputs["input_ids"][i]
        current_labels = list(current_input_ids) # Copy

        # Check length issue
        if input_len >= len(current_labels):
             logger.warning(f"Example {i}: Input part length ({input_len}) >= total length ({len(current_labels)}). Masking all labels.")
             final_labels.append([-100] * len(current_labels)) # Mask everything
             continue # Skip the normal masking loop

        # Normal masking
        for j in range(input_len):
             current_labels[j] = -100

        # Mask padding tokens
        try:
            for k in range(input_len, len(current_labels)):
                 if current_labels[k] == tokenizer.pad_token_id:
                      current_labels[k] = -100
        except Exception: pass

        final_labels.append(current_labels)


    # Ensure final_labels has the same number of entries as input_ids
    if len(model_inputs["input_ids"]) != len(final_labels):
         logger.error(f"FATAL: Mismatch between input_ids count ({len(model_inputs['input_ids'])}) and final_labels count ({len(final_labels)}) after processing batch.")
         # This indicates a bug in the processing logic above
         # As a fallback, create dummy labels, but this hides the error
         # final_labels = [[-100]*len(ids) for ids in model_inputs["input_ids"]] # Fallback - hides error
         raise RuntimeError("Label processing resulted in inconsistent list count.") # Better to fail

    model_inputs["labels"] = final_labels

    return model_inputs

logger.info("Tokenizing dataset (this might take a while)...")
tokenized_dataset = raw_dataset.map(
    format_and_tokenize,
    batched=True,
    remove_columns=raw_dataset.column_names, # Clean up dataset
    desc="Running tokenizer on dataset",
    # num_proc=os.cpu_count() // 2 # Optional: Speed up tokenization
)
logger.info("Tokenization complete.")
logger.info(f"Tokenized dataset structure: {tokenized_dataset}")

# --- Step 4: Load Qwen Model ---
logger.info(f"--- Loading Base Model: {model_name} ---")
# Determine compute dtype (bf16 is preferred if available)
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
logger.info(f"Using compute dtype: {compute_dtype}")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=compute_dtype, # Load in optimized dtype
        # device_map='auto', # Enable for multi-GPU or very large models
    )

    # **Crucial:** Configure the model's pad_token_id to match the tokenizer's
    # This prevents issues during training, especially with DataCollator/padding
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("Base model loaded and pad_token_id configured.")

except Exception as e:
    logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
    logger.error("Check model name, internet connection, and GPU memory.")
    exit(1)

# --- Step 5: Configure Training ---
logger.info("--- Configuring Training Arguments ---")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=paper_learning_rate,
    max_steps=paper_max_steps, # Train for a fixed number of steps
    # optim="adamw_torch", # Default Hugging Face AdamW
    adam_beta1=paper_adam_beta1,
    adam_beta2=paper_adam_beta2,
    adam_epsilon=paper_adam_epsilon,
    weight_decay=paper_weight_decay,
    lr_scheduler_type="linear", # Standard scheduler
    warmup_steps=100,           # Small number of warmup steps
    logging_strategy="steps",
    logging_steps=50,           # Log loss every 50 steps
    save_strategy="steps",
    save_steps=1000,            # Save checkpoint every 1000 steps
    save_total_limit=2,         # Only keep the last 2 checkpoints
    bf16=torch.cuda.is_bf16_supported(), # Use bf16 if available
    fp16=not torch.cuda.is_bf16_supported(), # Use fp16 if bf16 not available
    gradient_checkpointing=True,# Save VRAM by recomputing activations during backward pass
    report_to="none",           # Disable external reporting unless needed (e.g., "wandb")
    # remove_unused_columns=True, # Generally safe
)

# Data collator handles dynamic padding within each batch for efficiency
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset, # Use the processed dataset
    # eval_dataset=... # Add evaluation dataset here if you have one
    tokenizer=tokenizer,
    data_collator=data_collator,
)
logger.info("Trainer initialized.")

# --- Step 6: Start Training ---
logger.info("--- Starting SFT Training ---")
try:
    train_result = trainer.train()
    logger.info("--- Training Finished ---")

    # Log metrics & save model
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info(f"Saving final model locally to {output_dir}...")
    trainer.save_model(output_dir) # Saves everything needed (weights, config, tokenizer)
    logger.info("Final model saved successfully.")

except Exception as e:
    logger.error(f"error occurred during training: {e}", exc_info=True)
    exit(1)

#Step 7: Basic Verification
logger.info("--- Running Basic Verification ---")

# Define the output file path
verification_output_file = "output.txt"
logger.info(f"Verification outputs will be saved to: {verification_output_file}")

try:
    # Load the fine-tuned model from the local directory
    logger.info(f"Loading fine-tuned model from {output_dir} for verification...")
    sft_pipe = pipeline(
        "text-generation",
        model=output_dir,      # Load from the save directory
        tokenizer=output_dir,  # Load tokenizer from the same directory
        trust_remote_code=True,
        device=0 if torch.cuda.is_available() else -1 # Use GPU 0 if available, else CPU
    )
    logger.info("Verification pipeline loaded.")

    # Example simple prompts (what user might type *before* adding " Rephrase:")
    test_simple_prompts = [
        "a cute puppy playing",
        "a lonely robot on mars",
        "dragon breathing fire over a castle",
        "photorealistic portrait of an old fisherman",
        "steampunk cityscape at dawn"
    ]

    # Open the output file in write mode ('w')
    # Using 'with' ensures the file is properly closed even if errors occur
    with open(verification_output_file, 'w', encoding='utf-8') as f_out:
        logger.info(f"Opened {verification_output_file} for writing.")

        for simple_prompt in test_simple_prompts:
            # Format input as the model expects based on 'source' field in training data
            input_text = f"{simple_prompt} Rephrase:"
            logger.info(f"\nTesting with input: '{input_text}'")
            f_out.write(f"Input Prompt: {simple_prompt}\n") # Write input to file

            # Generate output
            try:
                result = sft_pipe(
                    input_text,
                    max_new_tokens=100, # Limit generation length
                    num_return_sequences=1,
                    do_sample=False,    # Use greedy decoding for predictable output
                    eos_token_id=tokenizer.eos_token_id, # Stop when EOS token is generated
                    pad_token_id=tokenizer.eos_token_id # Use EOS for padding during generation if needed
                )[0] # Get the first generated sequence

                generated_full_text = result['generated_text']
                # Extract only the part generated *after* the input prompt
                generated_output_parts = generated_full_text.split(input_text, 1)
                if len(generated_output_parts) > 1:
                    generated_output = generated_output_parts[1].strip()
                else:
                    # Handle case where model might just repeat input or generate nothing new
                    generated_output = "[Model did not generate text after the prompt]"

            except Exception as gen_e:
                logger.error(f"Error during generation for prompt '{simple_prompt}': {gen_e}")
                generated_output = "[Error during generation]"

            logger.info(f"Generated Output: '{generated_output}'")
            f_out.write(f"Generated Output: {generated_output}\n") # Write generated output
            f_out.write("-" * 30 + "\n") # Add a separator line

    logger.info(f"Verification outputs saved to {verification_output_file}")

except Exception as e:
    logger.error(f"Error during verification setup or execution: {e}", exc_info=True)

logger.info(" finished sft ")