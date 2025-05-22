import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

class AttackModel:
    def __init__(self):
        self.clf = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # Deep enough for 44-D input
            activation='relu',
            solver='adam',
            alpha=1e-4,          # L2 regularization to prevent overfitting
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            verbose=False
        )

    def train(self, features: np.ndarray, labels: np.ndarray):
        self.clf.fit(features, labels)

    def predict_proba(self, features: np.ndarray):
        return self.clf.predict_proba(features)[:, 1]

    def evaluate_auc(self, features: np.ndarray, labels: np.ndarray):
        scores = self.predict_proba(features)
        return roc_auc_score(labels, scores)
