# CLIP & HPS Score Evaluation Scripts

This project contains two evaluation scripts:
- `test_clip_score.py`: Uses CLIP model to evaluate image and prompt matching
- `test_hpsv2_score.py`: Uses HPSv2 model to evaluate image and prompt matching

## Output Description
### test_clip_score.py Output
- Generated images are saved in the output_dir directory
- Evaluation results are saved in rejected_evaluation_results.csv, containing the following fields:
  - original_prompt: Original prompt
  - rejected_prompt: Rejected prompt  
  - original_score: Original score
  - clip_score: CLIP score
  - rejected_clip_score: CLIP score for rejected prompt
  - image_path: Path to generated image

### test_hpsv2_score.py Output
- Adds two new columns to the input CSV file:
  - original_hps_score: HPS score for original prompt
  - rejected_hps_score: HPS score for rejected prompt

## Usage Steps
1. Move the script files from the folder to the main directory (just move them out of the folder)
2. Use the previously generated rejected_changed_prompt.csv file
3. Run test_clip_score.py to generate images (output to evaluation_output folder) and CLIP scores
4. Then run test_hpsv2_score.py to calculate HPS scores
5. All results will be saved in rejected_evaluation_results.csv file