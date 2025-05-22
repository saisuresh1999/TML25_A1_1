# TML25_A1_<YourTeamNumber>

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
