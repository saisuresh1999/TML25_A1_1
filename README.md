
# Membership Inference Attack (TML25_A1_1)

This project implements a **Membership Inference Attack (MIA)** against a target image classification model using a black-box threat model. The goal is to determine whether a specific data point was part of the training set of the target model, based only on its output behavior.

---

## Approach

We followed a **shadow training + score-based attack** pipeline enhanced by **Likelihood Ratio Attack (LiRA)** and a carefully tuned **Gradient Boosting Classifier** as the attack model.

### 1. Shadow Model Ensemble
We trained **7 shadow models** using disjoint subsets of the public data. Each shadow model mimics the target model's training behavior and is used to simulate membership/non-membership distributions.

- Architecture: ResNet18 
- Dataset: Membership-labeled public set (`pub.pt`)
- Normalization: Mean = [0.2980, 0.2962, 0.2987], Std = [0.2886, 0.2875, 0.2889]
- Training: 200 epochs with **early stopping** based on validation accuracy

### 2. Feature Extraction
For each sample in the shadow models’ test sets, we extracted:

- Softmax **confidence**
- **Entropy**
- **Margin** (Top-1 − Top-2 probability)

These features were then used to train the attack model.

### 3. LiRA (Likelihood Ratio Attack)
We saved the softmax **confidence values** for shadow member and non-member samples. These were modeled using Gaussian distributions. The attack score for each private sample is computed as:

```
P(member) = p_m / (p_m + p_nm)
```

Where `p_m` and `p_nm` are the PDF values under the member and non-member distributions respectively.

### 4. Attack Model
A **GradientBoostingClassifier** was trained on extracted features from shadow outputs:

- `n_estimators = 100`
- `learning_rate = 0.1`
- `max_depth = 3`
- `subsample = 0.9`
- Trained on confidence, entropy, and margin features
- Target samples were scored via LiRA for final submission

---

## Running the Pipeline

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train Shadow + Attack Models
```bash
python main.py
```
This performs shadow model training, feature extraction, and attack model training.

### Step 3: Generate Final Submission File
```bash
python submit.py
```
This loads the target model, extracts confidence values, applies LiRA, and creates `test.csv`.

---

## Folder Structure

```
TML25_A1_1/
├── attacks/
│   ├── attack_model.py         # Gradient Boosting attack classifier
│   └── feature_extractor.py   # Extracts confidence, entropy, margin
├── data/
│   └── dataset.py              # MembershipDataset and loaders
├── models/
│   └── shadow_model.py         # ResNet18 + early stopping
├── main.py                     # End-to-end shadow + attack pipeline
├── submit.py                   # LiRA inference and submission logic
├── pub.pt                      # Public dataset with labels
├── priv_out.pt                 # Private unlabeled dataset
├── shadow_model.pt             # Saved PyTorch model (shadow)
├── attack_model.pkl            # Trained attack classifier
├── member_conf.npy             # Shadow members’ confidence scores
├── nonmember_conf.npy          # Shadow non-members’ confidence scores
├── test.csv                    # Final submission file
├── config.py                   # Global paths and hyperparameters
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Important Implementation Highlights

- **Early stopping** is implemented inside `models/shadow_model.py` to avoid overfitting.
- All feature extraction logic resides in `attacks/feature_extractor.py`, which supports multiple statistics.
- **LiRA** logic is inside `submit.py`.
- Attack model trained using `GradientBoostingClassifier` in `attacks/attack_model.py`.

---

## Final Configuration

| Component           | Configuration                          |
|---------------------|----------------------------------------|
| Shadow Models       | 7 ResNet-18 models (200 epochs + ES)   |
| Attack Model        | GradientBoostingClassifier             |
| Feature Set         | Confidence, Entropy, Margin            |
| Submission Method   | LiRA (Likelihood Ratio using softmax)  |

```text
Best Result:
TPR@FPR=0.05 = 0.06166666666666667
AUC          = 0.5121307777777777
```

---

## Summary

This solution demonstrates that LiRA, combined with conservative shadow modeling and carefully chosen statistical features, can outperform naive classifier-based attacks on the TML25_A1_1 benchmark. Despite the subtle output differences of the target model, probabilistic modeling provided a robust advantage.
