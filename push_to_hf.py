from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

model_name = "AzalKhan/gpt2_DPO_01"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2_DPO_01")
model = AutoModelForCausalLM.from_pretrained("gpt2_DPO_01")

# Push to Hugging Face Hub
tokenizer.push_to_hub(model_name)
model.push_to_hub(model_name)
