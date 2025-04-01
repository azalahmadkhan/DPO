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

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load local model
local_model_path = "./gpt_01"  # Adjust this path to where your model files are stored
model_dpo_01 = AutoModelForCausalLM.from_pretrained(local_model_path, local_files_only=True).to(device)
tokenizer_dpo_01 = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
tokenizer_dpo_01.pad_token = tokenizer_dpo_01.eos_token
tokenizer_dpo_01.padding_side = "left"
model_dpo_01.eval()

def generate_prompt_dpo(plain_text):
    input_ids = tokenizer_dpo_01(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids.to(device)
    eos_id = tokenizer_dpo_01.eos_token_id
    outputs = model_dpo_01.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = tokenizer_dpo_01.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

print(generate_prompt_dpo("A beautiful image of a cat"))