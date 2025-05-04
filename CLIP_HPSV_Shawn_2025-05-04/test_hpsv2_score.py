import hpsv2
from PIL import Image
import pandas as pd
import os

# Read CSV file
csv_path = "/home/shawn/DPO/evaluation_output/rejected_evaluation_results.csv"
df = pd.read_csv(csv_path)
print(f"Read {len(df)} records")

# HPS version setting
hps_version = "v2.1"

# Process each row
for idx, row in df.iterrows():
    try:
        image_path = row['image_path']
        original_prompt = row['original_prompt']
        rejected_prompt = row['rejected_prompt']
        
        print(f"\nProcessing record {idx+1}/{len(df)}:")
        print(f"Image path: {image_path}")
        
        # Calculate HPS score for original prompt
        print("Calculating HPS score for original prompt...")
        original_hps_score = hpsv2.score(image_path, original_prompt, hps_version=hps_version)[0]
        print(f"Original prompt HPS score: {original_hps_score:.4f}")
        
        # Calculate HPS score for rejected prompt
        print("Calculating HPS score for rejected prompt...")
        rejected_hps_score = hpsv2.score(image_path, rejected_prompt, hps_version=hps_version)[0]
        print(f"Rejected prompt HPS score: {rejected_hps_score:.4f}")
        
        # Update DataFrame
        df.at[idx, 'original_hps_score'] = original_hps_score
        df.at[idx, 'rejected_hps_score'] = rejected_hps_score
        
        # Save updated results
        df.to_csv(csv_path, index=False)
        print(f"Saved record {idx+1}")
        
    except Exception as e:
        print(f"Error processing record {idx+1}: {e}")
        continue

print("\nAll data processing completed!")