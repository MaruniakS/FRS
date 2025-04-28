import tensorflow as tf
import shap
import numpy as np
import pandas as pd
import json
from constants import FEATURES

np.random.seed(42) 

SEQUENCE_LENGTH = 10
NUM_CLASSES = 3


def compute_shap_importance_per_class(model, X):
    num_samples = X.shape[0] // SEQUENCE_LENGTH
    X = X[:num_samples * SEQUENCE_LENGTH].reshape((num_samples, SEQUENCE_LENGTH, X.shape[1]))

    bg_indices = np.random.choice(len(X), size=10, replace=False)
    test_indices = np.random.choice(len(X), size=100, replace=False)

    background = X[bg_indices].reshape(10, -1)
    test_data = X[test_indices].reshape(100, -1)

    def model_fn(x):
        return model.predict(x.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)))

    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(test_data)  # List of [class_0, class_1, class_2], each (samples, features_flat)

    results = {}

    for class_index in range(NUM_CLASSES):
        class_shap = shap_values[class_index]  # shape: (samples, sequence_len * features)
        class_shap_seq = class_shap.reshape(-1, SEQUENCE_LENGTH, len(FEATURES))
        class_shap_flat = class_shap_seq.reshape(-1, len(FEATURES))  # (samples * sequence_len, features)

        # Raw mean absolute SHAP values
        mean_abs_per_feature = np.mean(np.abs(class_shap_flat), axis=0)
        raw = dict(zip(FEATURES, mean_abs_per_feature))

        # Normalized (sum = 1)
        total = mean_abs_per_feature.sum()
        normalized = {f: v / total for f, v in raw.items()}

        results[class_index] = normalized

    with open("class_importance.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    model = tf.keras.models.load_model("../out.model")
    df = pd.read_csv("../data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values

    per_class_importance = compute_shap_importance_per_class(model, X)
    print("Per-class SHAP feature importance saved to class_importance.json")
    print(json.dumps(per_class_importance, indent=4))
