import tensorflow as tf
import shap
import numpy as np
import pandas as pd
import json
from constants import FEATURES

np.random.seed(42) 

SEQUENCE_LENGTH = 10
NUM_CLASSES = 3


def compute_shap_signs(model, X):
    num_samples = X.shape[0] // SEQUENCE_LENGTH
    X = X[:num_samples * SEQUENCE_LENGTH].reshape((num_samples, SEQUENCE_LENGTH, X.shape[1]))

    bg_indices = np.random.choice(len(X), size=10, replace=False)
    test_indices = np.random.choice(len(X), size=100, replace=False)

    background = X[bg_indices].reshape(10, -1)
    test_data = X[test_indices].reshape(100, -1)

    def model_fn(x):
        return model.predict(x.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)))

    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(test_data)

    sign_results = {}

    for class_index in range(NUM_CLASSES):
        class_shap = shap_values[class_index]  # shape: (samples, features_flat)
        class_shap_seq = class_shap.reshape(-1, SEQUENCE_LENGTH, len(FEATURES))
        class_shap_flat = class_shap_seq.reshape(-1, len(FEATURES))

        # Sign: -1, 0, +1
        signs = np.sign(class_shap_flat)
        mean_sign = np.mean(signs, axis=0)

        sign_results[class_index] = dict(zip(FEATURES, mean_sign))

    with open("class_direction.json", "w") as f:
        json.dump(sign_results, f, indent=4)

    return sign_results


if __name__ == "__main__":
    model = tf.keras.models.load_model("../out.model")
    df = pd.read_csv("../data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values

    sign_summary = compute_shap_signs(model, X)
    print("Average SHAP direction (signs) saved to class_direction.json")
    print(json.dumps(sign_summary, indent=4))
