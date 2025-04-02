import pandas as pd
import random
from tqdm import tqdm

def read_preference_data(file_path):
    """Read preference_data.csv file"""
    print("Reading preference_data.csv...")
    df = pd.read_csv(file_path)
    return set(df['prompt'].tolist())

def read_filtered_data(file_path):
    """Read filtered_mix_train_for_ft.txt file"""
    print("Reading filtered_mix_train_for_ft.txt...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in tqdm(f)]

def select_unique_prompts(filtered_prompts, preference_prompts, num_samples=50000):
    """Select unique prompts"""
    print("Selecting unique prompts...")
    # Filter out prompts that already exist in preference_data
    unique_prompts = [p for p in filtered_prompts if p not in preference_prompts]
    
    # If there are not enough unique prompts, use all available prompts
    if len(unique_prompts) < num_samples:
        print(f"Warning: Available unique prompts ({len(unique_prompts)}) is less than requested amount ({num_samples})")
        selected_prompts = unique_prompts
    else:
        selected_prompts = random.sample(unique_prompts, num_samples)
    
    return selected_prompts

def save_to_csv(prompts, output_file):
    """Save selected prompts to CSV file"""
    print(f"Saving results to {output_file}...")
    df = pd.DataFrame({'prompt': prompts})
    df.to_csv(output_file, index=False)
    print(f"Saved {len(prompts)} prompts to {output_file}")

def main():
    # File paths
    preference_file = 'preference_data.csv'
    filtered_file = 'filtered_mix_train_for_ft.txt'
    output_file = 'selected_prompts.csv'
    
    # Read data
    preference_prompts = read_preference_data(preference_file)
    filtered_prompts = read_filtered_data(filtered_file)
    
    # Select unique prompts
    selected_prompts = select_unique_prompts(filtered_prompts, preference_prompts)
    
    # Save results
    save_to_csv(selected_prompts, output_file)

if __name__ == "__main__":
    main() 