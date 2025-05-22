# TML25_A1_<YourTeamNumber>

# Membership Inference Attack – TML Assignment 1

This repository contains our implementation for the Membership Inference Attack (MIA) assignment in the Trustworthy Machine Learning course (SS 2025).

## Problem Overview

Given a fixed ResNet18 target model and two datasets (`pub.pt` and `priv_out.pt`), the task is to predict the likelihood that each sample in `priv_out.pt` was part of the training set used for the target model. The challenge is evaluated based on TPR@FPR=0.05 and AUC.

## Our Approach

We implemented a classical shadow model-based MIA pipeline, with several extensions for robustness and ensemble diversity. Our final setup includes:

1. **Shadow Model Ensembling**:

   - We split the public dataset `pub.pt` into three disjoint subsets.
   - For each split, we trained a separate shadow model (ResNet18).
   - From each model, we extracted softmax outputs for the test split and labeled them with binary membership indicators.
2. **Feature Extraction**:

   - We used the full softmax vector (44 dimensions) as features for the attack model.
   - This representation allows the attack model to exploit class-wise distributional patterns instead of relying only on scalar summaries like entropy or confidence.
3. **Attack Model**:

   - A single MLP was trained on the aggregated softmax outputs from all shadow models.
   - This model learns to distinguish between member and non-member distributions by generalizing across shadow models.
4. **Submission Pipeline**:

   - For submission, the final shadow model was used to extract softmax features from `priv_out.pt`.
   - These features were passed to the trained attack model, which outputs the final membership scores.

## Directory Structure

# Membership Inference Attack – TML Assignment 1

This repository contains our implementation for the Membership Inference Attack (MIA) assignment in the Trustworthy Machine Learning course (SS 2025).

## Problem Overview

Given a fixed ResNet18 target model and two datasets (`pub.pt` and `priv_out.pt`), the task is to predict the likelihood that each sample in `priv_out.pt` was part of the training set used for the target model. The challenge is evaluated based on TPR@FPR=0.05 and AUC.

## Our Approach

We implemented a classical shadow model-based MIA pipeline, with several extensions for robustness and ensemble diversity. Our final setup includes:

1. **Shadow Model Ensembling**:

   - We split the public dataset `pub.pt` into three disjoint subsets.
   - For each split, we trained a separate shadow model (ResNet18).
   - From each model, we extracted softmax outputs for the test split and labeled them with binary membership indicators.
2. **Feature Extraction**:

   - We used the full softmax vector (44 dimensions) as features for the attack model.
   - This representation allows the attack model to exploit class-wise distributional patterns instead of relying only on scalar summaries like entropy or confidence.
3. **Attack Model**:

   - A single MLP was trained on the aggregated softmax outputs from all shadow models.
   - This model learns to distinguish between member and non-member distributions by generalizing across shadow models.
4. **Submission Pipeline**:

   - For submission, the final shadow model was used to extract softmax features from `priv_out.pt`.
   - These features were passed to the trained attack model, which outputs the final membership scores.

## Directory Structure

**Membership Inference Attack – Trustworthy Machine Learning (SS 2025)**
Authors: [Your Name], [Teammate Name]

## Task Overview

We implement a Membership Inference Attack (MIA) to determine whether a given input image was part of the training data of a provided ResNet18 model.

---

## Implementation Structure

```text
tml_2025_tasks/
├── data/
│   └── dataset.py              # Loads and splits pub.pt, loads priv_out.pt
├── models/
│   └── shadow_model.py         # ResNet18 training loop for shadow model
├── attacks/
│   ├── feature_extractor.py    # Extracts softmax-based features (confidence, entropy, margin)
│   └── attack_model.py         # Logistic Regression attack model
├── utils/
│   └── (Optional utility functions, unused)
├── main.py                     # Orchestrates shadow model training + attack training
├── submit.py                   # Loads trained models, predicts membership scores, submits
├── config.py                   # Global paths, batch sizes, token
├── shadow_model.pt             # Trained shadow model (ResNet18)
├── attack_model.pkl            # Trained logistic regression model
└── test.csv                    # Final membership score predictions for priv_out.pt
```
