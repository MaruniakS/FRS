import os
import tensorflow as tf
import shap
import numpy as np
import pandas as pd
import json
from constants import FEATURES

np.random.seed(42)

SEQUENCE_LENGTH = 10
NUM_CLASSES = 3

BG_SIZE = 20
TEST_SIZE = 200

RESULTS_DIR = "./results"
CLASS_IMPORTANCE_PATH = os.path.join(RESULTS_DIR, "class_importance.json")


def compute_shap_importance_per_class(model, X):
    num_samples = X.shape[0] // SEQUENCE_LENGTH
    X = X[:num_samples * SEQUENCE_LENGTH].reshape((num_samples, SEQUENCE_LENGTH, X.shape[1]))

    bg_indices = np.random.choice(len(X), size=BG_SIZE, replace=False)
    test_indices = np.random.choice(len(X), size=TEST_SIZE, replace=False)

    background = X[bg_indices].reshape(BG_SIZE, -1)
    test_data = X[test_indices].reshape(TEST_SIZE, -1)

    def model_fn(x):
        return model.predict(x.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)))

    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(test_data)  # list[class] each: (samples, seq_len * n_features)

    results = {}

    for class_index in range(NUM_CLASSES):
        class_shap = shap_values[class_index]  # (TEST_SIZE, SEQ_LEN * N_FEATURES)
        class_shap_seq = class_shap.reshape(-1, SEQUENCE_LENGTH, len(FEATURES))
        class_shap_flat = class_shap_seq.reshape(-1, len(FEATURES))  # (TEST_SIZE * SEQ_LEN, N_FEATURES)

        # Raw mean absolute SHAP values
        mean_abs_per_feature = np.mean(np.abs(class_shap_flat), axis=0)

        # Normalized (sum = 1)
        total = mean_abs_per_feature.sum()
        normalized = {f: float(v / total) for f, v in zip(FEATURES, mean_abs_per_feature)}

        results[class_index] = normalized

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CLASS_IMPORTANCE_PATH, "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    model = tf.keras.models.load_model("./out.model")
    df = pd.read_csv("./data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values

    per_class_importance = compute_shap_importance_per_class(model, X)
    print(f"Per-class SHAP feature importance saved to {CLASS_IMPORTANCE_PATH}")
    print(json.dumps(per_class_importance, indent=4))
