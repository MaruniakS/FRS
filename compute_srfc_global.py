import tensorflow as tf
import shap
import numpy as np
import pandas as pd
import json
from constants import FEATURES

SEQUENCE_LENGTH = 10


def compute_global_sfrc(model, X):
    num_samples = X.shape[0] // SEQUENCE_LENGTH
    X = X[:num_samples * SEQUENCE_LENGTH].reshape((num_samples, SEQUENCE_LENGTH, X.shape[1]))

    background = X[:10].reshape(10, -1)   # Flattened
    test_data = X[:100].reshape(100, -1)  # Flattened

    def model_fn(x):
        return model.predict(x.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)))

    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(test_data)

    # Combine SHAP values from all classes
    shap_all = np.sum(shap_values, axis=0)  # Shape: (samples, features_flat)
    shap_seq = shap_all.reshape(-1, SEQUENCE_LENGTH, len(FEATURES))
    shap_flat = shap_seq.reshape(-1, len(FEATURES))  # (samples * seq_len, features)

    # 1. Variance-based SFRC
    var_per_feature = np.var(shap_flat, axis=0)
    sfrc_var = 1 - (var_per_feature / var_per_feature.sum())

    # 2. Mean absolute SHAP SFRC
    mean_abs_per_feature = np.mean(np.abs(shap_flat), axis=0)
    sfrc_mean = 1 - (mean_abs_per_feature / mean_abs_per_feature.sum())

    sfrc_results = {
        "SFRC_variance": dict(zip(FEATURES, sfrc_var)),
        "SFRC_mean_abs": dict(zip(FEATURES, sfrc_mean))
    }

    with open("sfrc_global.json", "w") as f:
        json.dump(sfrc_results, f, indent=4)

    return sfrc_results


if __name__ == "__main__":
    model = tf.keras.models.load_model("out.model")
    df = pd.read_csv("data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values

    sfrc_global = compute_global_sfrc(model, X)
    print("Global SFRC coefficients saved to sfrc_global.json")
    print(json.dumps(sfrc_global, indent=4))
