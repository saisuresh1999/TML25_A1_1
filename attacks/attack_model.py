import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

class AttackModel:
    def __init__(self):
        self.clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.9,
            random_state=42
        )

    def train(self, features: np.ndarray, labels: np.ndarray):
        self.clf.fit(features, labels)

    def predict_proba(self, features: np.ndarray):
        return self.clf.predict_proba(features)[:, 1]

    def evaluate_auc(self, features: np.ndarray, labels: np.ndarray):
        scores = self.predict_proba(features)
        return roc_auc_score(labels, scores)
