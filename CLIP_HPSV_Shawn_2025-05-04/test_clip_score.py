import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import pandas as pd
import os
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
print("Loading models...")

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load Stable Diffusion
sd_model = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to(device)

def generate_image(prompt, output_path):
    """Generate image using Stable Diffusion"""
    if os.path.exists(output_path):
        print(f"Image exists, skipping generation")
        return Image.open(output_path)
        
    with torch.inference_mode():
        image = sd_model(
            prompt, 
            guidance_scale=7.5,
            num_inference_steps=20,
            height=512,
            width=512
        ).images[0]
    
    image.save(output_path)
    return image

def get_clip_score(image, prompt):
    """Calculate CLIP score"""
    if isinstance(image, str):
        image = Image.open(image)
        
    inputs = clip_processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
        clip_score = outputs.logits_per_image.mean().item()
    
    return clip_score

def evaluate_rejected_prompts(rejected_prompts_file):
    """Evaluate rejected prompts"""
    output_dir = "/home/shawn/DPO/evaluation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read prompts file
    df = pd.read_csv(rejected_prompts_file)
    print(f"Read {len(df)} prompts")
    
    results = []
    result_file = os.path.join(output_dir, "rejected_evaluation_results.csv")
    
    # Load existing results if any
    if os.path.exists(result_file):
        try:
            existing_results = pd.read_csv(result_file)
            results = existing_results.to_dict('records')
            print(f"Loaded {len(results)} existing results")
        except Exception as e:
            print(f"Failed to load existing results: {e}")
    
    # Process each prompt
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        if any(r.get('original_prompt') == row["original_prompt"] and 
               r.get('rejected_prompt') == row["rejected_prompt"] for r in results):
            print(f"Skipping processed prompt {idx}")
            continue
            
        try:
            original_prompt = row["original_prompt"]
            rejected_prompt = row["rejected_prompt"]
            original_score = row["score"]
            
            print(f"\nProcessing prompt {idx}:")
            print(f"Original prompt: {original_prompt}")
            print(f"Rejected prompt: {rejected_prompt}")
            
            # Generate image
            image_path = os.path.join(output_dir, f"evaluate_image_{idx}.png")
            generate_image(rejected_prompt, image_path)
            
            # Calculate scores
            print(f"Calculating CLIP score...")
            clip_score = get_clip_score(image_path, original_prompt)
            print(f"CLIP score: {clip_score}")
            
            print(f"Calculating rejected prompt CLIP score...")
            rejected_clip_score = get_clip_score(image_path, rejected_prompt)
            print(f"Rejected prompt CLIP score: {rejected_clip_score}")
            
            # Save result
            result = {
                "original_prompt": original_prompt,
                "rejected_prompt": rejected_prompt,
                "original_score": original_score,
                "clip_score": clip_score,
                "rejected_clip_score": rejected_clip_score,
                "image_path": image_path
            }
            results.append(result)
            
            # Save results periodically
            pd.DataFrame(results).to_csv(result_file, index=False)
            print(f"Saved {len(results)} results to {result_file}")
            
        except Exception as e:
            print(f"Error processing prompt {idx}: {str(e)}")
            continue
    
    # Final save
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(result_file, index=False)
        print("Evaluation completed!")
        print(f"Results saved to: {result_file}")
    else:
        print("No results to save!")

if __name__ == "__main__":
    rejected_prompts = "/home/shawn/DPO/test_prompts.csv"
    evaluate_rejected_prompts(rejected_prompts)