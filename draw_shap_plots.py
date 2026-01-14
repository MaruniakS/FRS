import os
import joblib
import numpy as np
import shap
import shap.plots
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from constants import FEATURES

SEQUENCE_LENGTH = 10
NUM_CLASSES = 3

BG_SIZE = 20
TEST_SIZE = 200

RESULTS_DIR = "./results"

SHAP_PKL_PATH = os.path.join(RESULTS_DIR, "shap_values.pkl")

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

def collapse_shap_time_global(shap_values, sequence_length, feature_count):
    # Sum SHAP values across classes
    if isinstance(shap_values, list):
        shap_all = np.sum(shap_values, axis=0)  # shape: (n_samples, sequence_length * n_features)
    else:
        shap_all = shap_values

    # Reshape to 3D: (samples, time, features)
    reshaped = shap_all.reshape(-1, sequence_length, feature_count)

    # Flatten time and samples into one axis
    flat = reshaped.reshape(-1, feature_count)

    # Compute global mean absolute SHAP per feature
    mean_abs = np.mean(np.abs(flat), axis=0)
    total = mean_abs.sum()
    normalized = mean_abs / total
    return normalized


def collapse_shap_time_per_class(shap_values, sequence_length, feature_count):
    """
    Returns a dictionary: {class_index: normalized_importance_vector}
    """
    class_importance = {}

    for class_index, class_shap in enumerate(shap_values):
        # Reshape: (samples, time, features)
        reshaped = class_shap.reshape(-1, sequence_length, feature_count)

        # Flatten time and samples into one axis
        flat = reshaped.reshape(-1, feature_count)

        # Compute mean absolute SHAP per feature
        mean_abs = np.mean(np.abs(flat), axis=0)
        total = mean_abs.sum()
        normalized = mean_abs / total
        class_importance[class_index] = normalized

    return class_importance  # dict[class_idx] = np.array of feature importances


def collapse_shap_direction_per_class(shap_values, sequence_length, feature_count):
    """
    Returns a dictionary: {class_index: directional_mean_vector}
    Each value is an array of length = number of features.
    """
    class_direction = {}

    for class_index, class_shap in enumerate(shap_values):
        # Reshape SHAP values to (samples, time, features)
        reshaped = class_shap.reshape(-1, sequence_length, feature_count)

        # Flatten over all time steps and samples
        flat = reshaped.reshape(-1, feature_count)

        # Compute directional sign (average sign across samples × time)
        signs = np.sign(flat)
        mean_sign = np.mean(signs, axis=0)

        class_direction[class_index] = mean_sign  # shape: (features,)

    return class_direction


def draw_global_summary_plot(shap_values):
    collapsed_shap = collapse_shap_time_global(
        shap_values=shap_values, sequence_length=SEQUENCE_LENGTH, feature_count=len(FEATURES),
    )

    explanation = shap.Explanation(
        values=collapsed_shap,
        feature_names=FEATURES
    )

    shap.plots.bar(explanation, max_display=17, show=True)


def draw_per_class_summary_plots(shap_values):
    class_importance = collapse_shap_time_per_class(
        shap_values,
        sequence_length=SEQUENCE_LENGTH,
        feature_count=len(FEATURES),
    )

    for class_index in range(NUM_CLASSES):
        collapsed_vals = class_importance[class_index]

        explanation = shap.Explanation(
            values=collapsed_vals,
            feature_names=FEATURES,
        )

        print(f"\n=== Class {class_index} ===")
        shap.plots.bar(explanation, max_display=17, show=True)


def draw_directional_summary_plots(shap_values):
    class_direction = collapse_shap_direction_per_class(
        shap_values,
        sequence_length=SEQUENCE_LENGTH,
        feature_count=len(FEATURES)
    )

    for class_index in range(NUM_CLASSES):
        directional_vals = class_direction[class_index]

        explanation = shap.Explanation(
            values=directional_vals,
            feature_names=FEATURES
        )

        print(f"\n=== SHAP Direction for Class {class_index} ===")
        shap.plots.bar(explanation, max_display=17, show=True)


def draw_global_beeswarm_plot(shap_values, test_data):
    if isinstance(shap_values, list):
        shap_all = np.sum(shap_values, axis=0)
    else:
        shap_all = shap_values

    reshaped_shap = shap_all.reshape(-1, SEQUENCE_LENGTH, len(FEATURES))
    flat_shap = reshaped_shap.reshape(-1, len(FEATURES))

    reshaped_input = test_data.reshape(-1, SEQUENCE_LENGTH, len(FEATURES))
    flat_input = reshaped_input.reshape(-1, len(FEATURES))

    explanation = shap.Explanation(
        values=flat_shap,
        # base_values=np.zeros(flat_shap.shape[0]),
        data=flat_input,
        feature_names=FEATURES
    )

    shap.plots.beeswarm(explanation, max_display=17, show=True)


def plot_shap_scatter_one_feature(shap_values, test_data, feature_name):
    feature_index = FEATURES.index(feature_name)

    # Prepare SHAP and input data
    shap_all = np.sum(shap_values, axis=0)
    flat_shap = shap_all.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)).reshape(-1, len(FEATURES))
    flat_input = test_data.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)).reshape(-1, len(FEATURES))

    # Plot SHAP dependence for one feature (scatter)
    shap.dependence_plot(
        ind=feature_index,
        shap_values=flat_shap,
        features=flat_input,
        feature_names=FEATURES,
        interaction_index=None  # disables coloring by interaction
    )


def plot_waterfall_for_feature(shap_values, test_data, sample_index, feature_name):
    shap_all = np.sum(shap_values, axis=0)
    flat_shap = shap_all.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)).reshape(-1, len(FEATURES))
    flat_input = test_data.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)).reshape(-1, len(FEATURES))

    expl = shap.Explanation(
        values=flat_shap[sample_index],
        base_values=0.0,
        data=flat_input[sample_index],
        feature_names=FEATURES
    )

    shap.plots.waterfall(expl)


def main():
    # Load SHAP values
    shap_values = joblib.load(SHAP_PKL_PATH)
    print("=== SHAP Values Loaded ===")

    df = pd.read_csv("./data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values
    X, _, test_data = prepare_data(X)

    draw_global_summary_plot(shap_values)
    draw_per_class_summary_plots(shap_values)
    draw_directional_summary_plots(shap_values)
    draw_global_beeswarm_plot(shap_values, test_data)
    plot_shap_scatter_one_feature(shap_values, test_data, "n_announcements")
    plot_shap_scatter_one_feature(shap_values, test_data, "av_number_of_bits_in_prefix_ipv4")



if __name__ == "__main__":
    main()

