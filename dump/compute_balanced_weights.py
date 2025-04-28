import tensorflow as tf
import shap
import numpy as np
import pandas as pd
from constants import FEATURES

SEQUENCE_LENGTH = 10  # Adjust based on your model's expected sequence length

def balance_weights(weights):
    """
    Balances SHAP weights using Min-Max Normalization.
    
    Parameters:
    - weights: Dictionary of feature SHAP weights.
    
    Returns:
    - Adjusted and normalized feature weights.
    """
    weight_values = np.array(list(weights.values()))
    
    # Min-Max Scaling
    min_val, max_val = np.min(weight_values), np.max(weight_values)
    weight_values = (weight_values - min_val) / (max_val - min_val + 1e-6)  # Avoid division by zero
    
    return dict(zip(weights.keys(), weight_values))

def compute_shap_weights(model, X, class_index):
    """
    Computes SHAP-based feature weights for a specific class using KernelExplainer.
    
    Parameters:
    - model: Trained TensorFlow model.
    - X: Input feature set (Pandas DataFrame or NumPy array).
    - class_index: Target class for SHAP weight computation.
    
    Returns:
    - Normalized feature weights as a dictionary.
    """
    # Ensure X has a shape of (batch_size, sequence_length, features)
    num_samples = X.shape[0] // SEQUENCE_LENGTH
    X = X[:num_samples * SEQUENCE_LENGTH].reshape((num_samples, SEQUENCE_LENGTH, X.shape[1]))

    # Select a subset of background samples for SHAP KernelExplainer
    background = X[:10].reshape(10, -1)  # Flatten input for KernelExplainer
    test_data = X[:100].reshape(100, -1)  # Flatten input for KernelExplainer

    # Initialize KernelExplainer
    explainer = shap.KernelExplainer(lambda x: model.predict(x.reshape(-1, SEQUENCE_LENGTH, len(FEATURES))), background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(test_data)
    shap_values_reshaped = [values.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)) for values in shap_values]

    # Aggregate per-class SHAP values
    per_class_impact = np.mean(np.abs(shap_values_reshaped[class_index]), axis=(0, 1))  # Mean over samples & time steps

    # Normalize weights using Min-Max Scaling
    raw_weights = dict(zip(FEATURES, per_class_impact))
    adjusted_weights = balance_weights(raw_weights)

    return adjusted_weights

if __name__ == "__main__":
    model = tf.keras.models.load_model("out.model")
    
    df = pd.read_csv("data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values
    class_index = 2  # Outage class
    
    weights = compute_shap_weights(model, X, class_index)
    print(weights)
