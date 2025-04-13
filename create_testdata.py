import pandas as pd
import numpy as np

# Set random seed to ensure reproducible results
np.random.seed(42)

# Read original CSV file
input_file = "/home/shawn/DPO/selected_prompts.csv"
output_file = "/home/shawn/DPO/testdata.csv"

try:
    # Read original prompt data
    df = pd.read_csv(input_file)
    
    # Check if data is sufficient
    if len(df) < 50:
        print(f"Warning: Original data only has {len(df)} rows, less than the requested 50 rows. Will use all available data.")
        sample_df = df
    else:
        # Randomly select 50 prompts
        sample_df = df.sample(n=50, random_state=42)
    
    # Save to testdata.csv
    sample_df.to_csv(output_file, index=False)
    
    print(f"Successfully randomly selected {len(sample_df)} prompts from {input_file}")
    print(f"Results saved to {output_file}")
    
except FileNotFoundError:
    print(f"Error: File not found {input_file}")
    print("Please ensure the file path is correct and the file exists.")
except Exception as e:
    print(f"An error occurred during processing: {str(e)}")