# Augmented_ReCo-RL

## Contents
- **ROCStories.csv**: This file contains the ROCStories dataset, comprising five-sentence stories.
- **roc_train.py**: A script used to fine-tune the pre-trained BERT model on the ROCStories dataset.
- **src/**: Modified src file originally copied from baseline.

## Baseline Model
Our work builds upon the baseline model ReCo-RL. The details and source code for ReCo-RL can be found at this [link](https://github.com/JunjieHu/ReCo-RL/tree/master). Due to GitHub's file size limitations, this repository only includes source (`src`) files copied from original baseline git repo and have been slightly modified for our project. For the rest of the files and detailed instructions on training and testing the model, please refer to the original ReCo-RL documentation and resources at the same link.

## Modifications to ReCo-RL

### Vist Model Adaptation
- The original Vist Model in ReCo-RL utilizes a pre-trained BERT model (`bert-uncased`) as its decoder.
- In our project, we have replaced this decoder with our ROC-trained model, which has been fine-tuned on the ROCStories dataset. This substitution aims to assess the impact of our tailored training on narrative generation.

### Source Code Changes
- **src/ Modifications**: 
  - `vist_model.py`: Line 609-613 has been altered to utilize our unimodal-trained BERT model as the decoder within the VIST model framework.
  - `scorer.py`: Line 18 (`Spice(), "SPICE"`) has been deleted due to version confliction.

## Training and Evaluation
First, run roc_train.py to get the pre-trained BERT model fine-tuned on ROCStories dataset.
To train and evaluate the modified model, please follow the instructions provided in the ReCo-RL documentation.
Useful commands:
Test: bash scripts/test.sh [GPU_id] (start from 0)
Outputs are saved at /outputs/rl/decode-beam5-len30.log
Train: 
- bash scripts/train_mle.sh [GPU_id] (start from 0)
- bash scripts/train_rl.sh [GPU_id]

## Notes:
Due to missing of training file in baseline models, only train_mle could be done at this stage. All the analysis are executed based on this result.
