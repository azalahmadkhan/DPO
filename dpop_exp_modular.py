import os
import csv
import torch
import hpsv2
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPProcessor,
    CLIPModel,
)
from diffusers import StableDiffusionPipeline, DDIMScheduler

# OPTIONAL: Future metric imports (as commentedout in the original code)
# from ImageReward import RM
# from pickscore import PickScoreModel
# from aesthetic_predictor import AestheticPredictor


# prompt model
class PromptModel:
    def __init__(self, model_name, tokenizer_name, device, local_files=False):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, local_files_only=local_files
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def generate_prompt(self, plain_text):
        input_text = f"{plain_text.strip()} Rephrase:"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        eos_id = self.tokenizer.eos_token_id
        outputs = self.model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=75,
            num_beams=8,
            num_return_sequences=8,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            length_penalty=-1.0,
        )
        output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_texts[0].replace(input_text, "").strip()


# image generator
class ImageGenerator:
    def __init__(self, model_id, device, output_folder="img"):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_auth_token=True
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def generate_image(self, prompt, filename):
        image = self.pipe(prompt, num_inference_steps=20).images[0]
        output_path = os.path.join(self.output_folder, filename)
        image.save(output_path)
        return output_path, image


# metrics
class HPSMetric:
    @staticmethod
    def compute(image_path, prompt):
        try:
            score = hpsv2.score(image_path, prompt, hps_version="v2.1")
            return float(score[0])
        except Exception as e:
            print(f"HPS computation error: {e}")
            return None

class CLIPMetric:
    def __init__(self, device):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    def compute_similarity(self, text1, text2):
        inputs = self.processor(
            text=[text1, text2], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return torch.nn.functional.cosine_similarity(features[0], features[1], dim=0).item()


# OPTIONAL: Additional metric placeholders (not active)
# class ImageRewardMetric:
#     def __init__(self, device):
#         self.model = RM.load("ImageReward-v1.0")
#     def compute(self, image_path, prompt):
#         return self.model.score(prompt, image_path)

# class AestheticsMetric:
#     def __init__(self):
#         self.model = AestheticPredictor("ava-hq")
#     def compute(self, image_path):
#         return self.model.predict(image_path)

# class PickScoreMetric:
#     def __init__(self):
#         self.model = PickScoreModel("ybelkada/pickscore_v1")
#     def compute(self, prompt, image_path):
#         return self.model.inference(prompt, image_path)


# evaluator
class DiffusionEvaluator:
    def __init__(self, input_csv, diffusion_output, device):
        self.input_csv = input_csv
        self.diffusion_output = diffusion_output
        self.device = device
        self.models = self._load_models()
        self.image_generator = ImageGenerator("CompVis/stable-diffusion-v1-4", device)
        self.metrics = {
            "hps": HPSMetric(),
            "clip": CLIPMetric(device),
            #optional metrics (optional, because they were commented out in the original code)
            #"image_reward": ImageRewardMetric(device),
            #"aesthetics": AestheticsMetric(),
            #"pickscore": PickScoreMetric(),
        }

    def _load_models(self):
        return {
            "sft": PromptModel("gpt2", "gpt2", self.device),
            "promptist": PromptModel("microsoft/Promptist", "microsoft/Promptist", self.device),
            "bloom": PromptModel(
                "alibaba-pai/pai-bloom-1b1-text2prompt-sd",
                "alibaba-pai/pai-bloom-1b1-text2prompt-sd",
                self.device,
            ),
            "dpo": PromptModel("AzalKhan/gpt2_dpo", "AzalKhan/gpt2_dpo", self.device),
            "dpo_10": PromptModel("./gpt_1", "./gpt_1", self.device, local_files=True),
            "dpo_20": PromptModel("./gpt_2", "./gpt_2", self.device, local_files=True),
            "dpo_30": PromptModel("./gpt_3", "./gpt_3", self.device, local_files=True),
            "dpo_50": PromptModel("./gpt_5", "./gpt_5", self.device, local_files=True),
        }

    def _read_prompts(self):
        with open(self.input_csv, "r") as f:
            return [row["prompt"] for row in csv.DictReader(f)]

    def _generate_all_prompts(self, base_prompt):
        return {
            "simple": base_prompt,
            **{name: model.generate_prompt(base_prompt) for name, model in self.models.items()}
        }

    def _process_single_prompt(self, idx, base_prompt):
        generated_prompts = self._generate_all_prompts(base_prompt)
        image_paths = []
        metric_data = {"hps": [], "clip": []}  # add keys for optional metrics as needed

        for model_name, prompt in generated_prompts.items():
            filename = f"{idx}_{model_name}.png"
            img_path, _ = self.image_generator.generate_image(prompt, filename)
            image_paths.append(img_path)

            metric_data["hps"].append(self.metrics["hps"].compute(img_path, prompt))

            if model_name != "simple":
                metric_data["clip"].append(
                    self.metrics["clip"].compute_similarity(base_prompt, prompt)
                )
            else:
                metric_data["clip"].append(1.0)

            #optionally: add to metric_data for each additional metric
            #metric_data["image_reward"].append(self.metrics["image_reward"].compute(img_path, prompt))
            #metric_data["aesthetics"].append(self.metrics["aesthetics"].compute(img_path))
            # etric_data["pickscore"].append(self.metrics["pickscore"].compute(prompt, img_path))

        for path in image_paths:
            os.remove(path)

        return metric_data

    def run_evaluation(self):
        prompts = self._read_prompts()
        with open(self.diffusion_output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "hps_simple", "hps_sft", "hps_promptist", "hps_bloom",
                "hps_dpo", "hps_dpo_10", "hps_dpo_20", "hps_dpo_30", "hps_dpo_50",
                "clip_simple", "clip_sft", "clip_promptist", "clip_bloom",
                "clip_dpo", "clip_dpo_10", "clip_dpo_20", "clip_dpo_30", "clip_dpo_50",
                # "image_reward_*", "aesthetics_*", "pickscore_*" column headers if activated
            ])

            for idx, prompt in enumerate(prompts):
                print(f"Processing prompt {idx+1}/{len(prompts)}")
                metrics = self._process_single_prompt(idx, prompt)
                row = metrics["hps"] + metrics["clip"]  # add other metric_data["..."] if active
                writer.writerow(row)


#main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = DiffusionEvaluator("diffusion.csv", "evaluation_results.csv", device)
    evaluator.run_evaluation()
