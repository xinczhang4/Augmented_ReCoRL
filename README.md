# Augmented_ReCo-RL

## Contents
- **ROCStories.csv**: This file contains the ROCStories dataset, comprising five-sentence stories.
- **roc_train.py**: A script used to fine-tune the pre-trained BERT model on the ROCStories dataset.
- **src/**: Modified src file originally copied from baseline.
- **decode-beam5-len30.tsv**: Output stories file for original ReCo-RL model.

## Baseline Model
Our work builds upon the baseline model ReCo-RL. The details and source code for ReCo-RL can be found at this [link](https://github.com/JunjieHu/ReCo-RL/tree/master). Due to GitHub's file size limitations, this repository only includes source (`src`) files copied from original baseline git repo and have been slightly modified for our project. For the rest of the files and detailed instructions on constructing the folder, please refer to the original ReCo-RL documentation and resources at the same link.

## Dataset Preparation
- RocStories dataset is already uploaded as ROCStories.csv.
- For ViST dataset, follow the [ReCO-RL](https://github.com/JunjieHu/ReCo-RL/tree/master) model to download to the correct folder.

## Modifications to ReCo-RL

### Vist Model Adaptation
- The original Vist Model in ReCo-RL utilizes a pre-trained BERT model (`bert-uncased`) as its decoder.
- In our project, we have replaced this decoder with our ROC-trained model, which has been fine-tuned on the ROCStories dataset. This substitution aims to assess the impact of our tailored training on narrative generation.

### Source Code Changes
- **src/ Modifications**: 
  - `vist_model.py`: Line 609-613 has been altered to utilize our unimodal-trained BERT model as the decoder within the VIST model framework.
  - `scorer.py`: Line 18 (`Spice(), "SPICE"`) has been deleted due to version confliction. New function self_bleu() (line 39 - 67) added to calculate the Self-BLEU score for each five-sentence story.

## Training and Evaluation
1. Run roc_train.py to get the pre-trained BERT model fine-tuned on ROCStories dataset, results will be saved in roc_trained/ folder.
2. To evaluate the baseline model, please follow the instructions provided in the ReCo-RL documentation and replace the src/ folder by ours.
Command: bash scripts/test.sh [GPU_id] (start from 0)
Outputs are saved at /outputs/rl/decode-beam5-len30.log and /outputs/rl/decode-beam5-len30.tsv(also uploaded here).
3. To train the baseline model:
- bash scripts/train_mle.sh [GPU_id] (start from 0)
- bash scripts/train_rl.sh [GPU_id] (start from 0)
3. To train the modified model, replace src/ folder with ours, and move the content in roc_trained/ folder to ReCo-RL/roc_trained:
- bash scripts/train_mle.sh [GPU_id] (start from 0)
- bash scripts/train_rl.sh [GPU_id] (currently not working because of original python file missing)

## Notes:
Due to missing of training file in baseline models, only train_mle could be done at this stage. Other analysis are done based on testing the original output from ReCo-RL model.
