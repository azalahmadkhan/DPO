import pandas as pd

# Read the original CSV file
df_original = pd.read_csv("human_preferences-100k.csv")

# Create a new DataFrame with columns 'prompt', 'accepted', and 'rejected'
df_new = pd.DataFrame(columns=['prompt', 'accepted', 'rejected'])

# Iterate through each row in the original DataFrame
for index, row in df_original.iterrows():
    if (index%1000==0):
        print(index)
    prompt = row['prompt']
    generated_prompt1 = row['generated_prompt1']
    generated_prompt2 = row['generated_prompt2']
    prompt1_score = row['prompt1_score']
    prompt2_score = row['prompt2_score']
    
    # Compare scores to determine which generated_prompt is accepted and which is rejected
    if prompt1_score > prompt2_score:
        accepted = generated_prompt1
        rejected = generated_prompt2
    else:
        accepted = generated_prompt2
        rejected = generated_prompt1
    
    # Append the data to the new DataFrame
    df_new = pd.concat([df_new, pd.DataFrame({'prompt': [prompt], 'accepted': [accepted], 'rejected': [rejected]})], ignore_index=True)

# Save the new DataFrame to a new CSV file
df_new.to_csv("preference_data.csv", index=False)
