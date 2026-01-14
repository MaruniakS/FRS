import os
import tensorflow as tf
import shap
import numpy as np
import pandas as pd
import json
from constants import FEATURES

np.random.seed(42)

SEQUENCE_LENGTH = 10

BG_SIZE = 20
TEST_SIZE = 200

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(RESULTS_DIR, "global_importance.json")


def compute_shap_importance(model, X):
    # Reshape data into (samples, sequence_len, features)
    num_samples = X.shape[0] // SEQUENCE_LENGTH
    X = X[:num_samples * SEQUENCE_LENGTH].reshape(
        (num_samples, SEQUENCE_LENGTH, X.shape[1])
    )

    # Flattened view for SHAP
    bg_indices = np.random.choice(len(X), size=BG_SIZE, replace=False)
    test_indices = np.random.choice(len(X), size=TEST_SIZE, replace=False)

    background = X[bg_indices].reshape(BG_SIZE, -1)
    test_data = X[test_indices].reshape(TEST_SIZE, -1)

    # Model wrapper for SHAP
    def model_fn(x):
        return model.predict(x.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)))

    # Use KernelExplainer for SHAP
    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(test_data)

    # Sum SHAP values over classes and reshape
    shap_all = np.sum(shap_values, axis=0)
    shap_seq = shap_all.reshape(-1, SEQUENCE_LENGTH, len(FEATURES))
    shap_flat = shap_seq.reshape(-1, len(FEATURES))

    # Mean absolute SHAP value per feature
    mean_abs_per_feature = np.mean(np.abs(shap_flat), axis=0)

    # Normalized version (sum = 1)
    total_importance = mean_abs_per_feature.sum()
    normalized_importance = {
        f: float(v / total_importance) for f, v in zip(FEATURES, mean_abs_per_feature)
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(normalized_importance, f, indent=4)

    return normalized_importance


if __name__ == "__main__":
    model = tf.keras.models.load_model("./out.model")
    df = pd.read_csv("./data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values

    global_importance = compute_shap_importance(model, X)
    print(f"SHAP feature importance saved to {OUTPUT_PATH}")
    print(json.dumps(global_importance, indent=4))
