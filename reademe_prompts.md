Required Installations
pip install transformers diffusers pandas pillow tqdm numpy
pip install git+https://github.com/THUDM/ImageReward.git

## File Description
- process_prompts.py : Main processing script
- selected_prompts.csv : Input prompt data
- Output files:
  - chosen_changed_prompt.csv : Selected optimized prompts
  - rejected_changed_prompt.csv : Rejected optimized prompts
  - output_images/ : Directory for generated images

## Path Configuration
When moving to a GPU server, modify the following paths:

1. Output directory:
```python
output_dir = "/home/shawn/DPO/output_images"
 ```
```

2. Result saving paths:
```python
chosen_df.to_csv("/home/shawn/DPO/chosen_changed_prompt.csv", index=False)
rejected_df.to_csv("/home/shawn/DPO/rejected_changed_prompt.csv", index=False)
 ```
```

3. Input CSV file path:
```python
input_csv = "/home/shawn/DPO/selected_prompts.csv"
 ```
```