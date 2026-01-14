#!/usr/bin/env python3
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Union

FILENAME = "frs-1-1-0.2.json"
FOLDERS = ["1", "2", "3", "4", "5"]
FIELDS = ["global", "class_avg", "dir_avg_abs", "frs"]


JsonType = Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]


def load_json(path: Path) -> JsonType:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_runs(data: JsonType) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list[dict]
      - list[list[dict]]
    Returns:
      - list[dict]
    """
    if not isinstance(data, list):
        raise TypeError(f"JSON root must be a list, got {type(data)}")

    if len(data) == 0:
        return []

    if all(isinstance(x, dict) for x in data):
        return data  # list[dict]

    # list[list[dict]] (or mixed)
    flat: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, list):
            for row in item:
                if not isinstance(row, dict):
                    raise TypeError(f"Expected dict inside nested list, got {type(row)}")
                flat.append(row)
        else:
            raise TypeError(f"Unsupported element type in JSON list: {type(item)}")
    return flat


def compute_feature_averages(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sums = defaultdict(lambda: {f: 0.0 for f in FIELDS})
    counts = defaultdict(int)

    for row in all_rows:
        if "feature" not in row:
            raise ValueError(f"Missing 'feature' key: {row}")
        feature = row["feature"]

        for f in FIELDS:
            if f not in row:
                raise ValueError(f"Missing '{f}' for feature='{feature}': {row}")
            sums[feature][f] += float(row[f])

        counts[feature] += 1

    result: List[Dict[str, Any]] = []
    for feature, s in sums.items():
        n = counts[feature]
        result.append(
            {
                "feature": feature,
                **{f: s[f] / n for f in FIELDS},
            }
        )

    # sort by avg frs desc
    result.sort(key=lambda r: r["frs"], reverse=True)
    return result


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, data: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", *FIELDS, "n"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def main() -> None:
    all_rows: List[Dict[str, Any]] = []

    for folder in FOLDERS:
        file_path = Path(folder) / FILENAME
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path.resolve()}")

        data = load_json(file_path)
        rows = flatten_runs(data)

        if not rows:
            print(f"WARNING: {file_path} is empty.")
            continue

        all_rows.extend(rows)
        print(f"Loaded {len(rows)} rows from {file_path}")

    averaged = compute_feature_averages(all_rows)

    # Print to stdout
    print(json.dumps(averaged, indent=2, ensure_ascii=False))

    # Save outputs
    save_json(Path("frs-averaged.json"), averaged)
    save_csv(Path("frs-averaged.csv"), averaged)

    print("\nSaved:")
    print(" - frs-averaged.json")
    print(" - frs-averaged.csv")


if __name__ == "__main__":
    main()
