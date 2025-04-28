import json

# === CONFIGURABLE WEIGHTS === #
ALPHA = 0.2  # weight for global importance
BETA = 0.2   # weight for class importance
GAMMA = 0.4  # weight for direction penalty

# === FILE PATHS === #
GLOBAL_FILE = "./global_importance.json"
CLASS_IMPORTANCE_FILE = "class_importance.json"
CLASS_DIRECTION_FILE = "class_direction.json"

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
    class_avg = sum(class_importance[str(cls)][feature] for cls in [0, 1, 2]) / 3
    direction_avg_abs = sum(abs(class_direction[str(cls)][feature]) for cls in [0, 1, 2]) / 3

    frs = ALPHA * global_val + BETA * class_avg - GAMMA * direction_avg_abs

    frs_results.append({
        "feature": feature,
        "global": round(global_val, 4),
        "class_avg": round(class_avg, 4),
        "dir_avg_abs": round(direction_avg_abs, 4),
        "frs": round(frs, 4)
    })

# === SORT RESULTS === #
frs_results_sorted = sorted(frs_results, key=lambda x: x["frs"])

# === PRINT TO CONSOLE === #
for row in frs_results_sorted:
    print(f"{row['feature']}: FRS = {row['frs']:.6f} (G={row['global']}, C={row['class_avg']}, D={row['dir_avg_abs']})")

# === SAVE TO FILE === #
output_file = f"frs-{ALPHA}-{BETA}-{GAMMA}.json"
with open(output_file, "w") as f:
    json.dump(frs_results_sorted, f, indent=4)

print(f"\nSaved FRS results to {output_file}")
