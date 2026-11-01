import json
import time
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from praxis import PRAXIS
from praxis import ThresholdGuessBinarizer
from run_standard_arborenum import DATASETS, load_dataset
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
LAMBDA_REG    = 0.02
DEPTH_BUDGET  = 5
RASHOMON_MULT = 0.03
LOOKAHEAD_K   = 1
RID_N_BOOT    = 3
RID_N_SCRAMBLE_EVALS = 5
RID_SEED      = 0
current = Path(__file__).resolve()
while current.name != "Threshold-Stability-and-Analysis":
    current = current.parent
BASEDIR = current
RESULTS_DIR = BASEDIR / "praxis_results"


def binarize(X_train, X_test, y_train):
    binarizer = ThresholdGuessBinarizer(random_state=RANDOM_SEED)
    binarizer.set_output(transform="pandas")
    X_train_bin = binarizer.fit_transform(X_train.astype(float), y_train)
    X_test_bin = pd.DataFrame(
        binarizer.transform(X_test.astype(float)),
        columns=X_train_bin.columns,
    )
    return binarizer, X_train_bin.reset_index(drop=True), X_test_bin


def export_praxis_feature_mapping(binarizer, out_dir):
    original_names = list(binarizer.feature_names_in_)
    output = {
        "n_original_features": int(binarizer.n_features_in_),
        "n_binarized_features": int(binarizer.n_features_out_),
        "binarized_features": [
            {
                "binarized_index": i,
                "name": name,
                "original_index": int(j),
                "original_feature": original_names[j],
                "threshold": float(thresh),
            }
            for i, ((j, thresh), name) in enumerate(
                zip(binarizer.thresholds_, binarizer.get_feature_names_out())
            )
        ],
        "feature_map": {
            original_names[j]: cols for j, cols in binarizer.feature_map().items()
        },
    }

    with open(out_dir / "praxis_feature_mapping.json", "w") as f:
        json.dump(output, f, indent=2)

    return output


def named_paths(paths_str, binarized_names):
    out = []
    for p in paths_str:
        conds = []
        for tok in p.strip("[]").split(", "):
            if not tok:
                continue
            name = binarized_names[int(tok[1:])]
            conds.append(name if tok[0] == "+" else f"NOT {name}")
        out.append("[" + ", ".join(conds) + "]")
    return out


def save_interval_inputs(model, X_train, X_train_bin, y_train, binarizer,
                         sampled_tree_indices, n_trees, out_dir):
    interval_dir = out_dir / "interval_inputs"
    interval_dir.mkdir(parents=True, exist_ok=True)
    X_train.reset_index(drop=True).to_csv(interval_dir / "X_train.csv", index=False)
    X_train_bin.to_csv(interval_dir / "X_train_binarized.csv", index=False)
    y_train.reset_index(drop=True).to_csv(interval_dir / "y_train.csv", index=False)
    binarized_names = list(binarizer.get_feature_names_out())
    exported_trees = []
    for idx in sampled_tree_indices:
        try:
            paths_str, leaf_preds = model.get_tree_paths_str(idx)
            raw_paths, raw_preds = model.get_tree_paths(idx)
            obj, obj_norm = model.get_tree_objective(idx)

            exported_trees.append({
                "tree_index": int(idx),
                "paths": list(paths_str),
                "named_paths": named_paths(paths_str, binarized_names),
                "predictions": [int(p) for p in leaf_preds],
                "objective": int(obj),
                "objective_normalized": float(obj_norm),
                "raw_paths_repr": repr(raw_paths),
                "raw_predictions_repr": repr(raw_preds),
            })
        except Exception as e:
            print(f"Interval export failed on tree {idx}: {e}")

    payload = {
        "n_total_trees": int(n_trees),
        "n_exported_trees": len(exported_trees),
        "complete_enumeration_export": len(exported_trees) == n_trees,
        "feature_names": list(X_train.columns),
        "binarized_feature_names": binarized_names,
        "thresholds": [
            {"original_index": int(j), "threshold": float(t)}
            for j, t in binarizer.thresholds_
        ],
        "parameters": {
            "lambda_reg": LAMBDA_REG,
            "depth_budget": DEPTH_BUDGET,
            "rashomon_mult": RASHOMON_MULT,
            "lookahead_k": LOOKAHEAD_K,
            "random_seed": RANDOM_SEED,
            "binarizer": "ThresholdGuessBinarizer",
            "binarizer_params": binarizer.get_params(),
        },
        "min_objective": float(model.get_min_objective()),
        "trees": exported_trees,
    }

    with open(interval_dir / "interval_inputs.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(exported_trees)} trees for interval-split analysis.")


def run_dataset(dataset_name, cfg):
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _expected_outputs = [
        "praxis_results.txt",
        "praxis_first_tree.txt",
        "praxis_tree_size.json",
        "praxis_sampled_trees.json",
        "praxis_first_tree_paths.txt",
        "praxis_rid.json",
        "interval_inputs/interval_inputs.json",
        "praxis_feature_mapping.json",
    ]
    if all((out_dir / f).exists() for f in _expected_outputs):
        print(f"  [PRAXIS] Skipping {dataset_name} — results already exist.")
        return

    X_train, X_test, y_train, y_test = load_dataset(cfg)
    print(f"  [PRAXIS] Binarizing {dataset_name} with ThresholdGuessBinarizer...")
    bin_start = time.perf_counter()
    binarizer, X_train_bin, X_test_bin = binarize(X_train, X_test, y_train)
    bin_duration = time.perf_counter() - bin_start
    print(f"  [PRAXIS] {X_train.shape[1]} features -> {X_train_bin.shape[1]} binary features")
    export_praxis_feature_mapping(binarizer, out_dir)
    binarized_names = list(X_train_bin.columns)
    model = PRAXIS()
    print(f"  [PRAXIS] Training on {dataset_name}...")
    start = time.perf_counter()
    model.fit(
        X_train_bin,
        y_train,
        lambda_reg=LAMBDA_REG,
        depth_budget=DEPTH_BUDGET,
        rashomon_mult=RASHOMON_MULT,
        lookahead_k=LOOKAHEAD_K,
    )
    duration = time.perf_counter() - start
    n_trees = model.count_trees()
    sampled_tree_indices = list(range(n_trees))
    tree_idx = 0   # Keep for evaluation and visualization
    test_preds = model.get_predictions(tree_idx, X_test_bin)
    acc = accuracy_score(y_test, test_preds)

    try:
        paths0, _ = model.get_tree_paths(tree_idx)
        n_leaves = len(paths0)
        n_nodes = 2 * n_leaves - 1
        tree_size = {
            "n_leaves": n_leaves,
            "n_nodes": n_nodes,
            "n_trees_in_set": n_trees,
            "n_sampled_trees": len(sampled_tree_indices),
        }
    except Exception as e:
        tree_size = {"error": str(e)}

    with open(out_dir / "praxis_tree_size.json", "w") as f:
        json.dump(tree_size, f, indent=2)

    all_tree_paths = []

    for idx in sampled_tree_indices:
        try:
            paths_str, leaf_preds = model.get_tree_paths_str(idx)

            all_tree_paths.append({
                "tree_index": idx,
                "paths": paths_str,
                "named_paths": named_paths(paths_str, binarized_names),
                "predictions": [int(p) for p in leaf_preds],
            })

        except Exception as e:
            print(f"Failed on tree {idx}: {e}")

    metadata = {
        "n_total_trees": n_trees,
        "n_sampled_trees": len(sampled_tree_indices),
        "trees": all_tree_paths,
    }

    with open(out_dir / "praxis_sampled_trees.json", "w") as f:
        json.dump(metadata, f, indent=2)

    save_interval_inputs(
        model=model,
        X_train=X_train,
        X_train_bin=X_train_bin,
        y_train=y_train,
        binarizer=binarizer,
        sampled_tree_indices=sampled_tree_indices,
        n_trees=n_trees,
        out_dir=out_dir,
    )

    with open(out_dir / "praxis_first_tree.txt", "w", encoding="utf-8") as fh:
        try:
            paths_str, tree_preds = model.get_tree_paths_str(tree_idx)
            fh.write("\n".join(f"{p} -> {pred}" for p, pred in zip(paths_str, tree_preds)))
        except Exception as e:
            fh.write(f"Tree unavailable: {e}\n")

    feature_names = list(X_train.columns)

    print(f"  [PRAXIS] Computing Rashomon Importance Distribution...")
    try:
        # Group binarized columns by original feature so RID reports importance per original feature
        binning_map = binarizer.feature_map()
        rid_start = time.perf_counter()
        rid_out = model.compute_rid(
            X_train_bin,
            y_train,
            n_boot=RID_N_BOOT,
            n_scramble_evals=RID_N_SCRAMBLE_EVALS,
            lambda_reg=LAMBDA_REG,
            depth_budget=DEPTH_BUDGET,
            rashomon_mult=RASHOMON_MULT,
            lookahead_k=LOOKAHEAD_K,
            seed=RID_SEED,
            memory_efficient=False,
            binning_map=binning_map,
        )
        rid_duration = time.perf_counter() - rid_start

        rid_feature_indices = sorted(binning_map.keys())
        with open(out_dir / "praxis_rid.json", "w") as f:
            json.dump(
                {
                    "feature_names": [feature_names[j] for j in rid_feature_indices],
                    "mean_sub_mr": [float(v) for v in rid_out["mean_sub_mr"]],
                },
                f,
            )

        rid_error = None
    except Exception as e:
        rid_duration = 0.0
        rid_error = str(e)


    print("Get Tree Paths:")
    try:
        paths_str, leaf_preds = model.get_tree_paths_str(tree_idx)

        with open(out_dir / "praxis_first_tree_paths.txt", "w", encoding="utf-8") as fh:
            for path, named, pred in zip(paths_str, named_paths(paths_str, binarized_names), leaf_preds):
                print(f"Path: {named} -> Prediction: {pred}")
                fh.write(f"{path} {named} -> {pred}\n")
    except Exception as e:
        print(f"Error retrieving tree paths: {e}")

    with open(out_dir / "praxis_results.txt", "w") as f:
        f.write(f"Accuracy: {acc}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, test_preds)}")
        f.write(f"\nThresholdGuessBinarizer: {X_train.shape[1]} features -> "
                f"{X_train_bin.shape[1]} binary features in {bin_duration:.2f} seconds")
        f.write(f"\nPRAXIS completed in {duration:.2f} seconds")
        f.write(f"\nRashomon set size: {n_trees}")
        f.write(f"\nMinimum objective: {model.get_min_objective()}")
        if "error" not in tree_size:
            f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")
        else:
            f.write(f"\nTree Size: Error - {tree_size['error']}")
        if rid_error is None:
            f.write(f"\nRID computed in {rid_duration:.2f} seconds (see praxis_rid.json)")
        else:
            f.write(f"\nRID: Error - {rid_error}")

    print(f"  [PRAXIS] Accuracy: {acc:.4f} | Time: {duration:.2f}s | Trees: {n_trees}")


if __name__ == "__main__":
    for dataset_name, cfg in DATASETS.items():
        print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        try:
            run_dataset(dataset_name, cfg)
        except Exception as e:
            print(f"  [PRAXIS] {dataset_name} failed: {e}")
