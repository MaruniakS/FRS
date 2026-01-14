import json
import os

# === CONFIGURABLE WEIGHTS === #
ALPHA = 1.0   # weight for global importance
BETA = 1.0    # weight for class importance
GAMMA = 0.2   # weight for direction penalty

# === PATHS === #
RESULTS_DIR = "./results"

GLOBAL_FILE = os.path.join(RESULTS_DIR, "global_importance.json")
CLASS_IMPORTANCE_FILE = os.path.join(RESULTS_DIR, "class_importance.json")
CLASS_DIRECTION_FILE = os.path.join(RESULTS_DIR, "class_direction.json")

OUTPUT_FILE = os.path.join(
    RESULTS_DIR, f"frs-{ALPHA}-{BETA}-{GAMMA}.json"
)

# === LOAD DATA === #
with open(GLOBAL_FILE, "r") as f:
    global_importance = json.load(f)

with open(CLASS_IMPORTANCE_FILE, "r") as f:
    class_importance = json.load(f)

with open(CLASS_DIRECTION_FILE, "r") as f:
    class_direction = json.load(f)

# === CALCULATE FRS === #
features = list(global_importance.keys())
frs_results = []

for feature in features:
    global_val = global_importance[feature]

    class_avg = sum(
        class_importance[str(cls)][feature] for cls in [0, 1, 2]
    ) / 3

    direction_avg_abs = sum(
        abs(class_direction[str(cls)][feature]) for cls in [0, 1, 2]
    ) / 3

    frs = ALPHA * global_val + BETA * class_avg - GAMMA * direction_avg_abs

    frs_results.append({
        "feature": feature,
        "global": round(global_val, 4),
        "class_avg": round(class_avg, 4),
        "dir_avg_abs": round(direction_avg_abs, 4),
        "frs": round(frs, 4)
    })

# === SORT RESULTS (DESCENDING FRS) === #
frs_results_sorted = sorted(
    frs_results, key=lambda x: x["frs"], reverse=True
)

# === PRINT TO CONSOLE === #
for row in frs_results_sorted:
    print(
        f"{row['feature']}: "
        f"FRS = {row['frs']:.6f} "
        f"(G={row['global']}, C={row['class_avg']}, D={row['dir_avg_abs']})"
    )

# === SAVE TO FILE === #
with open(OUTPUT_FILE, "w") as f:
    json.dump(frs_results_sorted, f, indent=4)

print(f"\nSaved FRS results to {OUTPUT_FILE}")
