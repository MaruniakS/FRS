import argparse
import tensorflow as tf
import shap
import numpy as np
import pandas as pd
import json
from constants import FEATURES
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

SEQUENCE_LENGTH = 10
NUM_CLASSES = 3

ALPHA = 1  # weight for global importance
BETA =  1  # weight for class importance
GAMMA = 0.2  # weight for direction penalty

GLOBAL_FILE = "global_importance200.json"
CLASS_IMPORTANCE_FILE = "class_importance200.json"
CLASS_DIRECTION_FILE = "class_direction200.json"
FRC_FILE = f"frs-{ALPHA}-{BETA}-{GAMMA}.json"

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


def compute_global_importance(shap_values):
    shap_all = np.sum(shap_values, axis=0)
    shap_seq = shap_all.reshape(-1, SEQUENCE_LENGTH, len(FEATURES))
    shap_flat = shap_seq.reshape(-1, len(FEATURES))
    mean_abs = np.mean(np.abs(shap_flat), axis=0)
    total = mean_abs.sum()
    normalized = {f: round(v / total, 4) for f, v in dict(zip(FEATURES, mean_abs)).items()}
    with open(GLOBAL_FILE, "w") as f:
        json.dump(normalized, f, indent=4)

    print("\n=== Global SHAP Importance (Sorted) ===")
    sorted_global = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, score) in enumerate(sorted_global, 1):
        print(f"{i:2}. {feature:35}: {score:.4f}")

    return normalized


def compute_class_importance(shap_values):
    results = {}
    for class_index in range(NUM_CLASSES):
        class_shap = shap_values[class_index].reshape(-1, SEQUENCE_LENGTH, len(FEATURES))
        flat = class_shap.reshape(-1, len(FEATURES))
        mean_abs = np.mean(np.abs(flat), axis=0)
        total = mean_abs.sum()
        normalized = {f: round(v / total, 4) for f, v in dict(zip(FEATURES, mean_abs)).items()}
        results[class_index] = normalized
    with open(CLASS_IMPORTANCE_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print("\n=== Per-Class SHAP Importance (Sorted) ===")
    for class_index in range(NUM_CLASSES):
        print(f"\nClass {class_index}:")
        sorted_class = sorted(results[class_index].items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_class, 1):
            print(f"{i:2}. {feature:35}: {score:.4f}")

    return results


def compute_class_direction(shap_values):
    signs_result = {}
    for class_index in range(NUM_CLASSES):
        class_shap = shap_values[class_index].reshape(-1, SEQUENCE_LENGTH, len(FEATURES))
        flat = class_shap.reshape(-1, len(FEATURES))
        signs = np.sign(flat)
        mean_sign = np.mean(signs, axis=0)
        signs_result[class_index] = dict(zip(FEATURES, mean_sign))
    with open(CLASS_DIRECTION_FILE, "w") as f:
        json.dump(signs_result, f, indent=4)
    print("\n=== Per-Class SHAP Direction (Sorted) ===")
    for class_index in range(NUM_CLASSES):
        print(f"\nClass {class_index}:")
        sorted_class = sorted(signs_result[class_index].items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_class, 1):
            print(f"{i:2}. {feature:35}: {score:.4f}")
    return signs_result


def compute_frc(global_importance, class_importance, class_direction):
    features = list(global_importance.keys())
    results = []
    for feature in features:
        global_val = global_importance[feature]
        class_avg = sum(class_importance[(cls)][feature] for cls in range(NUM_CLASSES)) / NUM_CLASSES
        dir_avg_abs = sum(abs(class_direction[(cls)][feature]) for cls in range(NUM_CLASSES)) / NUM_CLASSES
        frc = ALPHA * global_val + BETA * class_avg - GAMMA * dir_avg_abs
        results.append({
            "feature": feature,
            "global": round(global_val, 4),
            "class_avg": round(class_avg, 4),
            "dir_avg_abs": round(dir_avg_abs, 4),
            "frs": round(frc, 4)
        })
    results_sorted = sorted(results, key=lambda x: x["frs"], reverse=True)
    with open(FRC_FILE, "w") as f:
        json.dump(results_sorted, f, indent=4)
    print("\n=== FRC Ranking ===")
    for row in results_sorted:
        print(f"{row['feature']}: FRS = {row['frs']:.6f} (G={row['global']}, C={row['class_avg']}, D={row['dir_avg_abs']})")
    print(f"\nSaved FRS results to {FRC_FILE}")
    return results_sorted


def draw_plots(shap_values):
    """
    Plot global and per-class SHAP summary plots.

    Parameters:
        shap_values: List of SHAP arrays per class (shape: list of (n_samples, sequence_length * n_features)).
    """
    excluded_feature = "var_as_degree_in_paths"
    plot_features = [f for f in FEATURES if f != excluded_feature]
    n_features = len(FEATURES)

    # Reshape per class to (samples, time, features)
    reshaped = [sv.reshape(-1, SEQUENCE_LENGTH, n_features) for sv in shap_values]

    # === Global Impact ===
    global_vals = np.mean(np.abs(np.concatenate(reshaped, axis=0)), axis=0).mean(axis=0)
    global_vals_plot = [global_vals[FEATURES.index(f)] for f in plot_features]

    plt.figure(figsize=(12, 6))
    plt.bar(plot_features, global_vals_plot)
    plt.ylabel("Mean Absolute SHAP Value")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("global_importance.png")
    plt.close()

    # === Per-Class Impact ===
    for class_idx, class_vals in enumerate(reshaped):
        class_mean = np.mean(np.abs(class_vals), axis=0).mean(axis=0)
        class_vals_plot = [class_mean[FEATURES.index(f)] for f in plot_features]
        plt.figure(figsize=(12, 6))
        plt.bar(plot_features, class_vals_plot)
        plt.ylabel("Mean Absolute SHAP Value")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f"class_importance_{class_idx}.png")
        plt.close()


def plot_shap_direction_heatmap(signs_result, class_labels=None):
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(len(signs_result))]

    # Convert keys to strings if needed
    if isinstance(list(signs_result.keys())[0], int):
        signs_result = {str(k): v for k, v in signs_result.items()}

    # Build DataFrame
    data = pd.DataFrame.from_dict(signs_result, orient='index')
    data = data[FEATURES]
    data.index = class_labels
    heatmap_data = data.T  # shape: (features, classes)

    # Plot heatmap
    plt.figure(figsize=(10, len(FEATURES) * 0.4))
    sns.heatmap(
        heatmap_data, annot=True, center=0, cmap="RdYlGn", fmt=".2f",
        cbar_kws={'label': 'Mean SHAP Direction'}
    )
    plt.xlabel("Anomaly Class")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("direction_heatmap.png")
    plt.close()


def main():
    if args.only_frc:
        with open(GLOBAL_FILE, "r") as f:
            global_imp = json.load(f)
        with open(CLASS_IMPORTANCE_FILE, "r") as f:
            class_imp = json.load(f)
        with open(CLASS_DIRECTION_FILE, "r") as f:
            class_dir = json.load(f)
        compute_frc(global_imp, class_imp, class_dir)
    else:
        model = tf.keras.models.load_model("../out.model")
        df = pd.read_csv("../data/ready/train-val.features").fillna(0)
        X = df[FEATURES].values
        X, background, test_data = prepare_data(X)
        explainer = shap.KernelExplainer(model_wrapper(model), background)
        shap_values = explainer.shap_values(test_data)

        draw_plots(shap_values)

        global_imp = compute_global_importance(shap_values)
        class_imp = compute_class_importance(shap_values)
        class_dir = compute_class_direction(shap_values)
        plot_shap_direction_heatmap(class_dir, class_labels=["Direct", "Indirect", "Outage"])
        compute_frc(global_imp, class_imp, class_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-frc', action='store_true', help='Flag to consider only FRC calculation')
    args = parser.parse_args()
    main()
