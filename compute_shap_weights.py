import tensorflow as tf
import shap
import numpy as np
import pandas as pd
import json
from constants import FEATURES

SEQUENCE_LENGTH = 10  # Adjust based on your model's expected sequence length
NUM_CLASSES = 3  # Adjust based on the number of anomaly classes


def compute_raw_shap_values(model, X):
    """
    Computes raw SHAP values for all classes and stores them in a file.
    
    Parameters:
    - model: Trained TensorFlow model.
    - X: Input feature set (Pandas DataFrame or NumPy array).
    
    Returns:
    - Dictionary of raw SHAP values per class.
    """
    num_samples = X.shape[0] // SEQUENCE_LENGTH
    X = X[:num_samples * SEQUENCE_LENGTH].reshape((num_samples, SEQUENCE_LENGTH, X.shape[1]))
    
    background = X[:10].reshape(10, -1)  # Flatten input for KernelExplainer
    test_data = X[:100].reshape(100, -1)  # Flatten input for KernelExplainer

    explainer = shap.KernelExplainer(lambda x: model.predict(x.reshape(-1, SEQUENCE_LENGTH, len(FEATURES))), background)
    shap_values = explainer.shap_values(test_data)
    shap_values_reshaped = [values.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)) for values in shap_values]

    raw_shap_results = {}
    for class_index in range(NUM_CLASSES):
        per_class_impact = np.mean(np.abs(shap_values_reshaped[class_index]), axis=(0, 1))  # Mean over samples & time steps
        raw_shap_results[class_index] = dict(zip(FEATURES, per_class_impact))

    with open("raw_shap_values.json", "w") as f:
        json.dump(raw_shap_results, f, indent=4)
    
    return raw_shap_results


if __name__ == "__main__":
    model = tf.keras.models.load_model("out.model")
    
    df = pd.read_csv("data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values
    
    raw_shap_values = compute_raw_shap_values(model, X)
    print("Raw SHAP values saved to raw_shap_values.json")
    print(raw_shap_values)