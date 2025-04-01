import csv
import os
from transformers import AutoProcessor, AutoModel
from PIL import Image
import hpsv2 
import torch
import numpy as np
from torchvision import transforms
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os
import ImageReward as RM
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
import os
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import numpy as np
from PIL import Image
import glob
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_sft = AutoModelForCausalLM.from_pretrained("gpt2", state_dict=torch.load("sft_gpt.bin")).to(device)
tokenizer_sft = AutoTokenizer.from_pretrained("gpt2")           # Load the tokenizer
tokenizer_sft.pad_token = tokenizer_sft.eos_token
tokenizer_sft.padding_side = "left"
model_sft.eval()

model_promptist = AutoModelForCausalLM.from_pretrained("microsoft/Promptist").to(device)
tokenizer_promptist = AutoTokenizer.from_pretrained("microsoft/Promptist")
tokenizer_promptist.pad_token = tokenizer_promptist.eos_token
tokenizer_promptist.padding_side = "left"
model_promptist.eval()

model_bloom = AutoModelForCausalLM.from_pretrained('alibaba-pai/pai-bloom-1b1-text2prompt-sd').to(device)
tokenizer_bloom = AutoTokenizer.from_pretrained('alibaba-pai/pai-bloom-1b1-text2prompt-sd')
tokenizer_bloom.pad_token = tokenizer_bloom.eos_token
tokenizer_bloom.padding_side = "left"
model_bloom.eval()

model_dpo = AutoModelForCausalLM.from_pretrained("AzalKhan/gpt2_dpo").to(device)
tokenizer_dpo = AutoTokenizer.from_pretrained("AzalKhan/gpt2_dpo")
tokenizer_dpo.pad_token = tokenizer_dpo.eos_token
tokenizer_dpo.padding_side = "left"
model_dpo.eval()

# local_model_path_01 = "./gpt_01"  # Adjust this path to where your model files are stored
# model_dpo_01 = AutoModelForCausalLM.from_pretrained(local_model_path_01, local_files_only=True).to(device)
# tokenizer_dpo_01 = AutoTokenizer.from_pretrained(local_model_path_01, local_files_only=True)
# tokenizer_dpo_01.pad_token = tokenizer_dpo_01.eos_token
# tokenizer_dpo_01.padding_side = "left"
# model_dpo_01.eval()

# local_model_path_03 = "./gpt_03"  # Adjust this path to where your model files are stored
# model_dpo_03 = AutoModelForCausalLM.from_pretrained(local_model_path_03, local_files_only=True).to(device)
# tokenizer_dpo_03 = AutoTokenizer.from_pretrained(local_model_path_03, local_files_only=True)
# tokenizer_dpo_03.pad_token = tokenizer_dpo_03.eos_token
# tokenizer_dpo_03.padding_side = "left"
# model_dpo_03.eval()

# local_model_path_05 = "./gpt_05"  # Adjust this path to where your model files are stored
# model_dpo_05 = AutoModelForCausalLM.from_pretrained(local_model_path_05, local_files_only=True).to(device)
# tokenizer_dpo_05 = AutoTokenizer.from_pretrained(local_model_path_05, local_files_only=True)
# tokenizer_dpo_05.pad_token = tokenizer_dpo_05.eos_token
# tokenizer_dpo_05.padding_side = "left"
# model_dpo_05.eval()

# local_model_path_07 = "./gpt_07"  # Adjust this path to where your model files are stored
# model_dpo_07 = AutoModelForCausalLM.from_pretrained(local_model_path_07, local_files_only=True).to(device)
# tokenizer_dpo_07 = AutoTokenizer.from_pretrained(local_model_path_07, local_files_only=True)
# tokenizer_dpo_07.pad_token = tokenizer_dpo_07.eos_token
# tokenizer_dpo_07.padding_side = "left"
# model_dpo_07.eval()

# local_model_path_09 = "./gpt_09"  # Adjust this path to where your model files are stored
# model_dpo_09 = AutoModelForCausalLM.from_pretrained(local_model_path_09, local_files_only=True).to(device)
# tokenizer_dpo_09 = AutoTokenizer.from_pretrained(local_model_path_09, local_files_only=True)
# tokenizer_dpo_09.pad_token = tokenizer_dpo_09.eos_token
# tokenizer_dpo_09.padding_side = "left"
# model_dpo_09.eval()

local_model_path_10 = "./gpt_1"  # Adjust this path to where your model files are stored
model_dpo_10 = AutoModelForCausalLM.from_pretrained(local_model_path_10, local_files_only=True).to(device)
tokenizer_dpo_10 = AutoTokenizer.from_pretrained(local_model_path_10, local_files_only=True)
tokenizer_dpo_10.pad_token = tokenizer_dpo_10.eos_token
tokenizer_dpo_10.padding_side = "left"
model_dpo_10.eval()

local_model_path_20 = "./gpt_2"  # Adjust this path to where your model files are stored
model_dpo_20 = AutoModelForCausalLM.from_pretrained(local_model_path_20, local_files_only=True).to(device)
tokenizer_dpo_20 = AutoTokenizer.from_pretrained(local_model_path_20, local_files_only=True)
tokenizer_dpo_20.pad_token = tokenizer_dpo_20.eos_token
tokenizer_dpo_20.padding_side = "left"
model_dpo_20.eval()

local_model_path_30 = "./gpt_3"  # Adjust this path to where your model files are stored
model_dpo_30 = AutoModelForCausalLM.from_pretrained(local_model_path_30, local_files_only=True).to(device)
tokenizer_dpo_30 = AutoTokenizer.from_pretrained(local_model_path_30, local_files_only=True)
tokenizer_dpo_30.pad_token = tokenizer_dpo_30.eos_token
tokenizer_dpo_30.padding_side = "left"
model_dpo_30.eval()

local_model_path_50 = "./gpt_5"  # Adjust this path to where your model files are stored
model_dpo_50 = AutoModelForCausalLM.from_pretrained(local_model_path_50, local_files_only=True).to(device)
tokenizer_dpo_50 = AutoTokenizer.from_pretrained(local_model_path_50, local_files_only=True)
tokenizer_dpo_50.pad_token = tokenizer_dpo_50.eos_token
tokenizer_dpo_50.padding_side = "left"
model_dpo_50.eval()

def generate_prompt_sft(plain_text):
    input_ids = tokenizer_sft(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_sft.eos_token_id
    outputs = model_sft.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_sft.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_promptist(plain_text):
    input_ids = tokenizer_promptist(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_promptist.eos_token_id
    outputs = model_promptist.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_promptist.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_bloom(plain_text):
    input_ids = tokenizer_bloom(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_bloom.eos_token_id
    outputs = model_bloom.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_bloom.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_dpo(plain_text):
    input_ids = tokenizer_dpo(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo.eos_token_id
    outputs = model_dpo.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_dpo_01(plain_text):
    input_ids = tokenizer_dpo_01(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo_01.eos_token_id
    outputs = model_dpo_01.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo_01.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_dpo_03(plain_text):
    input_ids = tokenizer_dpo_03(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo_03.eos_token_id
    outputs = model_dpo_03.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo_03.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_dpo_05(plain_text):
    input_ids = tokenizer_dpo_05(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo_05.eos_token_id
    outputs = model_dpo_05.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo_05.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_dpo_07(plain_text):
    input_ids = tokenizer_dpo_07(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo_07.eos_token_id
    outputs = model_dpo_07.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo_07.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_dpo_09(plain_text):
    input_ids = tokenizer_dpo_09(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo_09.eos_token_id
    outputs = model_dpo_09.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo_09.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_dpo_10(plain_text):
    input_ids = tokenizer_dpo_10(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo_10.eos_token_id
    outputs = model_dpo_10.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo_10.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_dpo_20(plain_text):
    input_ids = tokenizer_dpo_20(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo_20.eos_token_id
    outputs = model_dpo_20.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo_20.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_dpo_30(plain_text):
    input_ids = tokenizer_dpo_30(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo_30.eos_token_id
    outputs = model_dpo_30.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo_30.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def generate_prompt_dpo_50(plain_text):
    input_ids = tokenizer_dpo_50(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo_50.eos_token_id
    outputs = model_dpo_50.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo_50.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

def calculate_text_similarity(text1, text2, clip_model, clip_processor):
    # Move model to specified device
    clip_model = clip_model.to(device)
    
    # Process both texts and move to device
    inputs1 = clip_processor(text=text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = clip_processor(text=text2, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to device
    inputs1 = {k: v.to(device) for k, v in inputs1.items()}
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}
    
    # Get text features
    with torch.no_grad():
        text_features1 = clip_model.get_text_features(**inputs1)
        text_features2 = clip_model.get_text_features(**inputs2)
    
    # Normalize features
    text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
    text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
    
    # Calculate similarity
    similarity = torch.nn.functional.cosine_similarity(text_features1, text_features2)
    return similarity.item()

model_id = "CompVis/stable-diffusion-v1-4"
output_folder = 'img'
os.makedirs(output_folder, exist_ok=True)
# Initialize the image generation pipeline on the GPU
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True, torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

# Initialize the image reward model
# model_rm = RM.load("ImageReward-v1.0").to(device)
# Initialize the clip model
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# Initialize the aesthetic model
# asth_extractor = AutoFeatureExtractor.from_pretrained("cafeai/cafe_aesthetic")
# asth_model = AutoModelForImageClassification.from_pretrained("cafeai/cafe_aesthetic")

import csv
import os
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

# # Load model for PickScore metric
# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

# processor = AutoProcessor.from_pretrained(processor_name_or_path)
# model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def calc_probs(prompt, images):
    # Preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # Embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # Score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        # Get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)

    return probs.cpu().tolist()

# Existing code
csv_file = 'diffusion.csv'
# Assuming the prompt column is named 'prompt'
prompts = []

with open(csv_file, 'r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        prompt = row['prompt']
        prompts.append(prompt)


image_reward = []
clips = []
aesthetics = []
pick_scores = []  # List to store PickScore metric scores
all_hps_scores = []   # List to store HPS metric scores
text_similarities = []  # Add this with other metric lists

# Generate text for each prompt
# for i, prompt in enumerate(prompts):
for i in range(len(prompts)):
    prompt = prompts[i]
    print(i)
    # Generate prompts from all models
    sft_prompt = generate_prompt_sft(prompt)
    promptist_prompt = generate_prompt_promptist(prompt)
    bloom_prompt = generate_prompt_bloom(prompt)
    dpo_prompt = generate_prompt_dpo(prompt)
    # dpo_01_prompt = generate_prompt_dpo_01(prompt)
    # dpo_03_prompt = generate_prompt_dpo_03(prompt)
    # dpo_05_prompt = generate_prompt_dpo_05(prompt)
    # dpo_07_prompt = generate_prompt_dpo_07(prompt)
    # dpo_09_prompt = generate_prompt_dpo_09(prompt)
    dpo_10_prompt = generate_prompt_dpo_10(prompt)
    dpo_20_prompt = generate_prompt_dpo_20(prompt)
    dpo_30_prompt = generate_prompt_dpo_30(prompt)
    dpo_50_prompt = generate_prompt_dpo_50(prompt)


    # Calculate text similarities for all models
    # clip_model.to(device)
    # text_similarities.append([
    #     1.0,  # original prompt
    #     calculate_text_similarity(prompt, sft_prompt, clip_model, clip_processor),
    #     calculate_text_similarity(prompt, promptist_prompt, clip_model, clip_processor),
    #     calculate_text_similarity(prompt, bloom_prompt, clip_model, clip_processor),
    #     # calculate_text_similarity(prompt, dpo_prompt, clip_model, clip_processor),
    #     # calculate_text_similarity(prompt, dpo_01_prompt, clip_model, clip_processor),
    #     # calculate_text_similarity(prompt, dpo_03_prompt, clip_model, clip_processor),
    #     # calculate_text_similarity(prompt, dpo_05_prompt, clip_model, clip_processor),
    #     # calculate_text_similarity(prompt, dpo_07_prompt, clip_model, clip_processor),
    #     # calculate_text_similarity(prompt, dpo_09_prompt, clip_model, clip_processor),
    #     calculate_text_similarity(prompt, dpo_10_prompt, clip_model, clip_processor),
    #     calculate_text_similarity(prompt, dpo_20_prompt, clip_model, clip_processor),
    #     calculate_text_similarity(prompt, dpo_30_prompt, clip_model, clip_processor),
    #     calculate_text_similarity(prompt, dpo_50_prompt, clip_model, clip_processor)

    # ])

    # Generate images for all prompts
    generated_prompts = [
        prompt, sft_prompt, promptist_prompt, bloom_prompt,
        dpo_prompt, 
        # dpo_01_prompt, dpo_03_prompt,
        # dpo_05_prompt, dpo_07_prompt, dpo_09_prompt
        dpo_10_prompt, dpo_20_prompt, dpo_30_prompt, dpo_50_prompt
    ]
    output_paths = []
    clip_scores = []
    asth_scores = []
    hps_scores = []
    images = []

    for j, p in enumerate(generated_prompts):
        output_path = os.path.join(output_folder, f'{i}{j}.png')
        output_paths.append(output_path)
        image = pipe(p, num_inference_steps=20).images[0]
        image.save(output_path)
        images.append(image)

        # Load the image using PIL
        image = Image.open(output_path)
        image = transforms.ToTensor()(image)  # Convert to PyTorch tensor
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(device)  # Move image to the same device as the model weights

        # Calculate CLIP score for the generated image
        # inputs = clip_processor(text=p, images=image, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        # clip_model.to(device)
        # with torch.no_grad():
        #     outputs = clip_model(**inputs)
        #     logits_per_image = outputs.logits_per_image
        #     clip_score = logits_per_image.mean().item()  # Get the mean score
        # print(clip_score)
        # clip_scores.append(clip_score)

        # Calculate HPS score
        try:
            hps_score = hpsv2.score(output_path, prompt, hps_version="v2.1")
            hps_score = float(hps_score[0])
            print(hps_score)
            hps_scores.append(hps_score)
        except Exception as e:
            print(f"Error with image {output_path}: {e}")
            hps_scores.append(None)


        # outputs = clip_model(**inputs)
        # logits_per_image = outputs.logits_per_image
        # clip_scores.append(logits_per_image.softmax(dim=1)[0][0].item())  # Extract and store score

        # im_features = asth_extractor(image).data['pixel_values']
        # if isinstance(im_features, list):
        #     im_features = np.array(im_features)
        # im_features = torch.from_numpy(im_features).to(device)  # Move to same device
        # asth_model.to(device)  # Move aesthetic model to same device
        # logits = asth_model(im_features)
        # probs = torch.nn.functional.softmax(logits.logits, dim=-1)
        # aesthetic_score = probs[0][-1].item()
        # asth_scores.append(aesthetic_score)

    # rewards = model_rm.score(prompt, output_paths)
    # pick_score = calc_probs(prompt, images)  # Calculate the PickScore for the current set of images

    # clips.append(clip_scores)
    # aesthetics.append(asth_scores)
    # image_reward.append(rewards)
    # pick_scores.append(pick_score)  # Append the PickScore
    # Store HPS scores
    all_hps_scores.append(hps_scores)  # Add this list at the top of the file: all_hps_scores = []

    # Clean up generated images
    for path in output_paths:
        os.remove(path)

# Open the CSV file in write mode
with open("diffusion_output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Update header row with all models
    writer.writerow([
        # "clip_simple", "clip_sft", "clip_promptist", "clip_bloom",
        # "clip_dpo", "clip_dpo_01", "clip_dpo_03", "clip_dpo_05",
        # "clip_dpo_07", "clip_dpo_09",
        # "clip_dpo_10", "clip_dpo_20", 
        # "clip_dpo_30", "clip_dpo_50",
        
        # "aesthetics_simple", "aesthetics_sft", "aesthetics_promptist",
        # "aesthetics_bloom", "aesthetics_dpo", "aesthetics_dpo_01",
        # "aesthetics_dpo_03", "aesthetics_dpo_05", "aesthetics_dpo_07",
        # "aesthetics_dpo_09", 
        # "aesthetics_dpo_10", "aesthetics_dpo_20",
        # "aesthetics_dpo_30", "aesthetics_dpo_50",
        
        # "ir_simple", "ir_sft", "ir_promptist", "ir_bloom",
        # "ir_dpo", "ir_dpo_01", "ir_dpo_03", "ir_dpo_05",
        # "ir_dpo_07", "ir_dpo_09", 
        # "ir_dpo_10", "ir_dpo_20",
        # "ir_dpo_30", "ir_dpo_50",
        
        # "pick_simple", "pick_sft", "pick_promptist", "pick_bloom",
        # "pick_dpo", "pick_dpo_01", "pick_dpo_03", "pick_dpo_05",
        # "pick_dpo_07", "pick_dpo_09", 
        # "pick_dpo_10", "pick_dpo_20",
        # "pick_dpo_30", "pick_dpo_50",
        
        # "text_sim_simple", "text_sim_sft", "text_sim_promptist",
        # "text_sim_bloom", "text_sim_dpo", "text_sim_dpo_01",
        # "text_sim_dpo_03", "text_sim_dpo_05", "text_sim_dpo_07",
        # "text_sim_dpo_09"
        # "text_dpo_10", "text_dpo_20",
        # "text_dpo_30", "text_dpo_50"

        "hps_simple", "hps_sft", "hps_promptist", "hps_bloom",
        "hps_dpo", 
        # "hps_dpo_01", "hps_dpo_02", "hps_dpo_03",
        # "hps_dpo_04", "hps_dpo_05", "hps_dpo_07", "hps_dpo_09",
        "hps_dpo_10", "hps_dpo_20", "hps_dpo_30", "hps_dpo_50",
    
    ])

    # Write data rows
    for i in range(len(prompts)):
        # Unpack all scores
        # clip_scores = clips[i]
        # aesthetic_scores = aesthetics[i]
        # ir_scores = image_reward[i]
        # pick_scores_row = pick_scores[i]
        # text_sim_scores = text_similarities[i]
        hps = all_hps_scores[i]

        # Write all scores to the row
        writer.writerow(
            # clip_scores 
            hps
            # aesthetic_scores +
            # ir_scores +
            # pick_scores_row +
            # text_sim_scores
        )