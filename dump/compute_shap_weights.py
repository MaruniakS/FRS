import tensorflow as tf
import shap
import numpy as np
import pandas as pd
from constants import FEATURES

SEQUENCE_LENGTH = 10  # Adjust based on your model's expected sequence length

def compute_shap_weights(model, X, class_index):
    """
    Computes SHAP-based feature weights for a specific class.
    
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

    # Select a background sample for SHAP
    background = X[np.random.choice(X.shape[0], 50, replace=False)]
    explainer = shap.GradientExplainer(model, background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X)
    
    # Extract SHAP values for the specified class
    class_shap_values = np.abs(shap_values[class_index]).mean(axis=(0, 1))  # Mean over samples & sequence length
    
    # Normalize weights
    weights = class_shap_values / np.sum(class_shap_values)

    return dict(zip(FEATURES, weights))

if __name__ == "__main__":
    model = tf.keras.models.load_model("out.model")
    
    df = pd.read_csv("data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values
    class_index = 2  # Set the target class index
    
    weights = compute_shap_weights(model, X, class_index)
    print(weights)
