import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import pandas as pd
import os
from tqdm import tqdm
import hpsv2  # 直接导入 hpsv2

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型
print("Loading models...")
# 加载CLIP模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 加载Stable Diffusion时启用内存优化
sd_model = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    enable_attention_slicing=True,  # 添加注意力切片
    enable_model_cpu_offload=True   # 启用CPU卸载
).to(device)

def generate_image(prompt, output_path):
    """使用Stable Diffusion生成图像"""
    with torch.autocast(device.type):
        # 添加内存清理
        with torch.cuda.amp.autocast():
            image = sd_model(
                prompt, 
                guidance_scale=7.5,
                num_inference_steps=30  # 减少推理步数
            ).images[0]
    
    image.save(output_path)
    torch.cuda.empty_cache()  # 清理GPU缓存
    return image

def evaluate_rejected_prompts(rejected_prompts_file):
    """评估被拒绝的提示词"""
    # 创建输出目录
    output_dir = "/home/shawn/DPO/evaluation_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取提示词文件
    df = pd.read_csv(rejected_prompts_file)
    
    # 创建结果DataFrame
    results = []
    
    # 处理每个提示词
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        try:
            original_prompt = row["original_prompt"]
            rejected_prompt = row["rejected_prompt"]
            original_score = row["score"]
            
            # 生成图像
            image_path = os.path.join(output_dir, f"evaluate_image_{idx}.png")
            image = generate_image(rejected_prompt, image_path)
            
            # 计算分数
            clip_score = get_clip_score(image, original_prompt)
            hps_score = get_hps_score(image, original_prompt)
            
            # 保存结果
            results.append({
                "original_prompt": original_prompt,
                "rejected_prompt": rejected_prompt,
                "original_score": original_score,
                "clip_score": clip_score,
                "hps_score": hps_score,
                "image_path": image_path
            })
            
            # 清理内存
            del image
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing prompt {idx}: {str(e)}")
            continue
            
        if (idx + 1) % 5 == 0:  # 改为每5个保存一次
            pd.DataFrame(results).to_csv(
                os.path.join(output_dir, "rejected_evaluation_results.csv"),
                index=False
            )
    
    # 最终保存
    final_df = pd.DataFrame(results)
    final_df.to_csv(
        os.path.join(output_dir, "rejected_evaluation_results.csv"),
        index=False
    )
    
    print("Evaluation complete!")
    print(f"Results saved to: {os.path.join(output_dir, 'rejected_evaluation_results.csv')}")

if __name__ == "__main__":
    rejected_prompts = "/home/shawn/DPO/rejected_changed_prompt.csv"
    evaluate_rejected_prompts(rejected_prompts)