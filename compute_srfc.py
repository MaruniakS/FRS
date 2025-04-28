import tensorflow as tf
import shap
import numpy as np
import pandas as pd
import json
from constants import FEATURES

SEQUENCE_LENGTH = 10
NUM_CLASSES = 3


def compute_sfrc(model, X):
    num_samples = X.shape[0] // SEQUENCE_LENGTH
    X = X[:num_samples * SEQUENCE_LENGTH].reshape((num_samples, SEQUENCE_LENGTH, X.shape[1]))
    
    background = X[:10].reshape(10, -1)   # Flattened
    test_data = X[:100].reshape(100, -1)  # Flattened

    # Wrapper to reshape back into model input shape
    def model_fn(x):
        return model.predict(x.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)))

    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(test_data)
    shap_values_reshaped = [values.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)) for values in shap_values]

    sfrc_results = {}

    for class_index in range(NUM_CLASSES):
        values = shap_values_reshaped[class_index]  # shape: (samples, sequence_len, features)

        # Flatten over samples and timesteps: (samples * seq_len, features)
        values_flat = values.reshape(-1, len(FEATURES))

        # 1. Variance-based SFRC
        var_per_feature = np.var(values_flat, axis=0)
        sfrc_var = 1 - (var_per_feature / var_per_feature.sum())

        # 2. Mean absolute SHAP SFRC
        mean_abs_per_feature = np.mean(np.abs(values_flat), axis=0)
        sfrc_mean = 1 - (mean_abs_per_feature / mean_abs_per_feature.sum())

        sfrc_results[class_index] = {
            "SFRC_variance": dict(zip(FEATURES, sfrc_var)),
            "SFRC_mean_abs": dict(zip(FEATURES, sfrc_mean))
        }

    with open("sfrc_coefficients.json", "w") as f:
        json.dump(sfrc_results, f, indent=4)

    return sfrc_results


if __name__ == "__main__":
    model = tf.keras.models.load_model("out.model")

    df = pd.read_csv("data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values

    sfrc_coefficients = compute_sfrc(model, X)
    print("SFRC coefficients saved to sfrc_coefficients.json")
    print(json.dumps(sfrc_coefficients, indent=4))
