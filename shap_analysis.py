import argparse
import os
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

ALPHA = 1
BETA = 1
GAMMA = 0.2

BG_SIZE = 20
TEST_SIZE = 200

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GLOBAL_FILE = os.path.join(RESULTS_DIR, "global_importance.json")
CLASS_IMPORTANCE_FILE = os.path.join(RESULTS_DIR, "class_importance.json")
CLASS_DIRECTION_FILE = os.path.join(RESULTS_DIR, "class_direction.json")
FRC_FILE = os.path.join(RESULTS_DIR, f"frs-{ALPHA}-{BETA}-{GAMMA}.json")


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
    normalized = {f: round(v / total, 4) for f, v in zip(FEATURES, mean_abs)}

    with open(GLOBAL_FILE, "w") as f:
        json.dump(normalized, f, indent=4)

    return normalized


def compute_class_importance(shap_values):
    results = {}
    for cls in range(NUM_CLASSES):
        flat = shap_values[cls].reshape(-1, SEQUENCE_LENGTH, len(FEATURES)).reshape(-1, len(FEATURES))
        mean_abs = np.mean(np.abs(flat), axis=0)
        total = mean_abs.sum()
        results[cls] = {f: round(v / total, 4) for f, v in zip(FEATURES, mean_abs)}

    with open(CLASS_IMPORTANCE_FILE, "w") as f:
        json.dump(results, f, indent=4)

    return results


def compute_class_direction(shap_values):
    signs_result = {}
    for cls in range(NUM_CLASSES):
        flat = shap_values[cls].reshape(-1, SEQUENCE_LENGTH, len(FEATURES)).reshape(-1, len(FEATURES))
        signs_result[cls] = dict(zip(FEATURES, np.mean(np.sign(flat), axis=0)))

    with open(CLASS_DIRECTION_FILE, "w") as f:
        json.dump(signs_result, f, indent=4)

    return signs_result


def compute_frc(global_imp, class_imp, class_dir):
    results = []
    for f in FEATURES:
        g = global_imp[f]
        c = sum(class_imp[i][f] for i in range(NUM_CLASSES)) / NUM_CLASSES
        d = sum(abs(class_dir[i][f]) for i in range(NUM_CLASSES)) / NUM_CLASSES
        frs = ALPHA * g + BETA * c - GAMMA * d

        results.append({
            "feature": f,
            "global": round(g, 4),
            "class_avg": round(c, 4),
            "dir_avg_abs": round(d, 4),
            "frs": round(frs, 4)
        })

    results.sort(key=lambda x: x["frs"], reverse=True)

    with open(FRC_FILE, "w") as f:
        json.dump(results, f, indent=4)

    return results

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
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "global_importance.png"))
    plt.close()

    # === Per-Class Impact ===
    for class_idx, class_vals in enumerate(reshaped):
        class_mean = np.mean(np.abs(class_vals), axis=0).mean(axis=0)
        class_vals_plot = [class_mean[FEATURES.index(f)] for f in plot_features]

        plt.figure(figsize=(12, 6))
        plt.bar(plot_features, class_vals_plot)
        plt.ylabel("Mean Absolute SHAP Value")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"class_importance_{class_idx}.png"))
        plt.close()


def draw_plots(shap_values):
    reshaped = [sv.reshape(-1, SEQUENCE_LENGTH, len(FEATURES)) for sv in shap_values]

    global_vals = np.mean(np.abs(np.concatenate(reshaped, axis=0)), axis=0).mean(axis=0)

    plt.figure(figsize=(12, 6))
    plt.bar(FEATURES, global_vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "global_importance.png"))
    plt.close()

    for i, cls_vals in enumerate(reshaped):
        vals = np.mean(np.abs(cls_vals), axis=0).mean(axis=0)
        plt.figure(figsize=(12, 6))
        plt.bar(FEATURES, vals)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean |SHAP|")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"class_importance_{i}.png"))
        plt.close()


def plot_shap_direction_heatmap(signs_result, class_labels):
    df = pd.DataFrame.from_dict(signs_result, orient="index")[FEATURES]
    df.index = class_labels

    plt.figure(figsize=(10, len(FEATURES) * 0.4))
    sns.heatmap(df.T, annot=True, center=0, cmap="RdYlGn", fmt=".2f")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "direction_heatmap.png"))
    plt.close()


def main():
    if args.only_frc:
        with open(GLOBAL_FILE) as f:
            global_imp = json.load(f)
        with open(CLASS_IMPORTANCE_FILE) as f:
            class_imp = json.load(f)
        with open(CLASS_DIRECTION_FILE) as f:
            class_dir = json.load(f)

        compute_frc(global_imp, class_imp, class_dir)
        return

    model = tf.keras.models.load_model("./out.model")
    df = pd.read_csv("./data/ready/train-val.features").fillna(0)
    X = df[FEATURES].values

    _, background, test_data = prepare_data(X)
    explainer = shap.KernelExplainer(model_wrapper(model), background)
    shap_values = explainer.shap_values(test_data)

    draw_plots(shap_values)

    global_imp = compute_global_importance(shap_values)
    class_imp = compute_class_importance(shap_values)
    class_dir = compute_class_direction(shap_values)

    plot_shap_direction_heatmap(class_dir, ["Direct", "Indirect", "Outage"])
    compute_frc(global_imp, class_imp, class_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-frc', action='store_true', help='Flag to consider only FRC calculation')
    args = parser.parse_args()
    main()
