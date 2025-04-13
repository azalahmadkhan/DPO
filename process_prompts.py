import os
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
import ImageReward as RM
import numpy as np
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
output_dir = "/home/shawn/DPO/output_images"
os.makedirs(output_dir, exist_ok=True)

# Load GPT-2 model for generating optimized prompts
print("Loading GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Load Stable Diffusion model
print("Loading Stable Diffusion model...")
sd_model = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)

# Load ImageReward model
print("Loading ImageReward model...")
# Use the correct model name
reward_model = RM.load("ImageReward-v1.0").to(device)

def generate_optimized_prompts(original_prompt, num_prompts=2):
    """Generate optimized prompts using GPT-2"""
    input_text = f"Original prompt: {original_prompt}\nOptimized prompt:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    optimized_prompts = []
    for _ in range(num_prompts):
        output = gpt2_model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 100,
            num_return_sequences=1,
            temperature=0.9,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        optimized_prompt = generated_text.split("Optimized prompt:")[1].strip()
        optimized_prompts.append(optimized_prompt)
    
    return optimized_prompts

def generate_image(prompt, image_path):
    """Generate image using Stable Diffusion"""
    with torch.autocast("cuda" if device.type == "cuda" else "cpu"):
        image = sd_model(prompt, guidance_scale=7.5).images[0]
    
    image.save(image_path)
    return image

def score_image(image, prompt):
    """Score image using ImageReward"""
    if isinstance(image, str):
        image = Image.open(image)
    
    with torch.no_grad():
        reward_score = reward_model.score(prompt, image)
    
    # Return reward_score directly, without calling .item()
    # because reward_score is already a float type
    return reward_score

def process_dataset(csv_path):
    """Process dataset and generate results"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create result dataframes
    chosen_df = pd.DataFrame(columns=["original_prompt", "chosen_prompt", "score"])
    rejected_df = pd.DataFrame(columns=["original_prompt", "rejected_prompt", "score"])
    
    # Process each prompt
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        original_prompt = row["prompt"]  # Assume the column name in CSV is "prompt"
        
        print(f"\nProcessing prompt {idx+1}/{len(df)}: {original_prompt[:50]}...")
        
        # Generate two optimized prompts
        optimized_prompts = generate_optimized_prompts(original_prompt)
        print(f"Optimized prompt 1: {optimized_prompts[0][:50]}...")
        print(f"Optimized prompt 2: {optimized_prompts[1][:50]}...")
        
        # Generate images for each optimized prompt
        image_paths = []
        for i, opt_prompt in enumerate(optimized_prompts):
            image_path = os.path.join(output_dir, f"prompt_{idx}_opt_{i}.png")
            print(f"Generating image for optimized prompt {i+1}...")
            generate_image(opt_prompt, image_path)
            image_paths.append(image_path)
        
        # Score images
        scores = []
        for i, (opt_prompt, img_path) in enumerate(zip(optimized_prompts, image_paths)):
            print(f"Scoring image for optimized prompt {i+1}...")
            score = score_image(img_path, opt_prompt)
            scores.append(score)
            print(f"Score for optimized prompt {i+1}: {score}")
        
        # Determine which prompt has higher score
        if scores[0] > scores[1]:
            chosen_idx, rejected_idx = 0, 1
        else:
            chosen_idx, rejected_idx = 1, 0
        
        # Add to result dataframes
        chosen_df = pd.concat([chosen_df, pd.DataFrame({
            "original_prompt": [original_prompt],
            "chosen_prompt": [optimized_prompts[chosen_idx]],
            "score": [scores[chosen_idx]]
        })], ignore_index=True)
        
        rejected_df = pd.concat([rejected_df, pd.DataFrame({
            "original_prompt": [original_prompt],
            "rejected_prompt": [optimized_prompts[rejected_idx]],
            "score": [scores[rejected_idx]]
        })], ignore_index=True)
    
    # Save results
    chosen_df.to_csv("/home/shawn/DPO/chosen_changed_prompt.csv", index=False)
    rejected_df.to_csv("/home/shawn/DPO/rejected_changed_prompt.csv", index=False)
    
    print("\nProcessing complete!")
    print(f"Chosen prompts saved to: /home/shawn/DPO/chosen_changed_prompt.csv")
    print(f"Rejected prompts saved to: /home/shawn/DPO/rejected_changed_prompt.csv")

if __name__ == "__main__":
    # Process testdata.csv as original_prompt.csv
    input_csv = "/home/shawn/DPO/testdata.csv"
    process_dataset(input_csv)