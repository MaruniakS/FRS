import joblib
import tensorflow as tf
import shap
import numpy as np
import pandas as pd
from constants import FEATURES

np.random.seed(42)

SEQUENCE_LENGTH = 10
NUM_CLASSES = 3

BG_SIZE = 20
TEST_SIZE = 200

def prepare_data(X):
    num_samples = X.shape[0] // SEQUENCE_LENGTH
    X = X[:num_samples * SEQUENCE_LENGTH].reshape((num_samples, SEQUENCE_LENGTH, X.shape[1]))
    bg_indices = np.random.choice(len(X), size=BG_SIZE, replace=False)
    test_indices = np.random.choice(len(X), size=TEST_SIZE, replace=False)
    background = X[bg_indices].reshape(BG_SIZE, -1)
    test_data = X[test_indices].reshape(TEST_SIZE, -1)
    return X, background, test_data


def model_wrapper(model):
    return lambda x: model.predict(x.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)))


def main():
    model = tf.keras.models.load_model("../out.model")
    df = pd.read_csv("../data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values
    X, background, test_data = prepare_data(X)
    explainer = shap.KernelExplainer(model_wrapper(model), background)
    shap_values = explainer.shap_values(test_data)
    joblib.dump(shap_values, "shap_values.pkl")


if __name__ == "__main__":
    main()
