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

def compute_shap_importance(model, X):
    # Reshape data into (samples, sequence_len, features)
    num_samples = X.shape[0] // SEQUENCE_LENGTH
    X = X[:num_samples * SEQUENCE_LENGTH].reshape((num_samples, SEQUENCE_LENGTH, X.shape[1]))

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
    shap_importance = dict(zip(FEATURES, mean_abs_per_feature))

    # Normalized version
    total_importance = mean_abs_per_feature.sum()
    normalized_importance = {
        f: v / total_importance for f, v in shap_importance.items()
    }

    with open(f"global_importance-{TEST_SIZE}-{BG_SIZE}.json", "w") as f:
        json.dump(normalized_importance, f, indent=4)

    return normalized_importance


if __name__ == "__main__":
    model = tf.keras.models.load_model("../out.model")
    df = pd.read_csv("../data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values

    global_importance = compute_shap_importance(model, X)
    print("SHAP feature importance saved to global_importance.json")
    print(json.dumps(global_importance, indent=4))
