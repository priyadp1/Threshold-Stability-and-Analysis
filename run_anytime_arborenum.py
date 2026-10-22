import inspect
import json
import time
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from arborenum import ArborEnum
from arborenum_bounds import capture_native_stdout, parse_bounds
import random

MAX_TREE_SAMPLE = 1000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
LAMBDA_REG    = 0.005
DEPTH_BUDGET  = 5
RASHOMON_MULT = 0.01
LOOKAHEAD_K   = 1
RID_N_BOOT    = 3
RID_N_SCRAMBLE_EVALS = 5
RID_SEED      = 0
SECOND_RASHOMON_MULT = 0.03
RUNTIME_LIMIT_SECONDS = 600
MEMORY_LIMIT_MB = 16000   # 24 GB machine; leave headroom for the OS
current = Path(__file__).resolve()
while current.name != "Threshold-Stability-and-Analysis":
    current = current.parent
BASEDIR = current
DATA_DIR    = BASEDIR / "DatasetWork"
RESULTS_DIR = BASEDIR / "anytime_arborenum_results"
DATASETS = {
    "abalone": {
        "path": DATA_DIR / "abalone_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "adult": {
        "path": DATA_DIR / "adult_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "aging": {
        "path": DATA_DIR / "aging_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "bank": {
        "path": DATA_DIR / "bank_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "bike": {
        "path": DATA_DIR / "bike_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "california": {
        "path": DATA_DIR / "california_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "christine": {
        "path": DATA_DIR / "christine_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "churn": {
        "path": DATA_DIR / "churn_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "compas": {
        "path": DATA_DIR / "compas_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "coupon": {
        "path": DATA_DIR / "coupon_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "credit": {
        "path": DATA_DIR / "credit_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "diabetes": {
        "path": DATA_DIR / "diabetes_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "diamonds": {
        "path": DATA_DIR / "diamonds_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "droid": {
        "path": DATA_DIR / "droid_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "helena": {
        "path": DATA_DIR / "helena_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "heloc": {
        "path": DATA_DIR / "heloc_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "jasmine": {
        "path": DATA_DIR / "jasmine_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "madeline": {
        "path": DATA_DIR / "madeline_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "magic": {
        "path": DATA_DIR / "magic_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "monk2": {
        "path": DATA_DIR / "monk2_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "news": {
        "path": DATA_DIR / "news_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "phishing": {
        "path": DATA_DIR / "phishing_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "pol": {
        "path": DATA_DIR / "pol_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "rl": {
        "path": DATA_DIR / "rl_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "shopping": {
        "path": DATA_DIR / "shopping_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "spambase": {
        "path": DATA_DIR / "spambase_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "student": {
        "path": DATA_DIR / "student_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "tictactoe": {
        "path": DATA_DIR / "tictactoe_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
    "wine": {
        "path": DATA_DIR / "wine_processed.csv",
        "target_col": "label",
        "drop_cols": ["label"],
        "label_map": None,
    },
}


def load_dataset(cfg):
    df = pd.read_csv(cfg["path"])
    if cfg.get("na_strategy") == "drop_columns":
        df = df.drop(columns=cfg["cols_to_drop_for_na"])
    else:
        df = df.dropna(axis=1, how="all")
        df = df.dropna()

    if cfg["label_map"]:
        df[cfg["target_col"]] = df[cfg["target_col"]].map(cfg["label_map"])

    X = df.drop(columns=cfg["drop_cols"])
    y = df[cfg["target_col"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def print_split_diagnostics(model, out_dir):
    # Inspect ArborEnum's split metadata to decode signed split IDs (+4, -27, ...)
    lines = [f"Model type: {type(model)}", "", "Available model attributes:"]
    for name in dir(model):
        if any(
            word in name.lower()
            for word in ["threshold", "feature", "split", "mapping", "encoder"]
        ):
            lines.append(name)

    lines += ["", "Raw tree paths (tree 0):"]
    try:
        raw_paths, raw_preds = model.get_tree_paths(0)
        lines.append(repr(raw_paths))
        lines.append(repr(raw_preds))
    except Exception as e:
        lines.append(f"Raw tree paths unavailable: {e}")

    text = "\n".join(lines)
    print(text)
    with open(out_dir / "split_id_diagnostics.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def export_arborenum_feature_mapping(model, out_dir):
    # Dump ArborEnum's internal feature/threshold metadata so split IDs can be decoded
    methods = [
        "get_continuous_threshold_info",
        "get_internal_feature_info",
        "get_internal_feature_names",
    ]

    output = {}
    for name in methods:
        method = getattr(model, name, None)
        if method is None:
            output[name] = {"status": "error", "error": "method not found"}
            continue

        try:
            signature = str(inspect.signature(method))
        except (TypeError, ValueError):
            signature = "unavailable"

        try:
            result = method()
            output[name] = {
                "signature": signature,
                "result": repr(result),
                "status": "success",
            }
        except Exception as e:
            output[name] = {
                "signature": signature,
                "error": str(e),
                "status": "error",
            }

    for attr in ["continuous_feature_indices_", "binary_feature_specs_"]:
        output[attr.rstrip("_")] = repr(getattr(model, attr, "attribute not found"))

    with open(out_dir / "arborenum_feature_mapping.json", "w") as f:
        json.dump(output, f, indent=2)

    return output


def fit_model(X_train, y_train):
    # Capture ArborEnum's C++ log so the Rashomon cutoff it actually used can be saved
    model = ArborEnum()
    with capture_native_stdout() as log:
        model.fit(
            X_train,
            y_train,
            proxy_mode = "hybrid",
            early_stopping=True,
            lambda_reg = LAMBDA_REG,
            depth_budget = DEPTH_BUDGET,
            rashomon_mult= RASHOMON_MULT,
            lookahead_k = LOOKAHEAD_K,
            max_number_thresholds_per_feature=None,
            key_mode="hash",
            proxy_refinement="auto" ,
            second_rashomon_mult=SECOND_RASHOMON_MULT,
            multiplier_step_size=0.0025,
            runtime_limit_seconds=RUNTIME_LIMIT_SECONDS,
            memory_limit_mb=MEMORY_LIMIT_MB
        )
    return model, parse_bounds(log["text"])


def backfill_bounds(out_dir):
    # Older runs did not save the cutoff; refit on the saved training data and patch it in.
    # With a runtime limit the refit may stop at a different budget, so require the same set size.
    interval_dir = out_dir / "interval_inputs"
    with open(interval_dir / "interval_inputs.json") as f:
        payload = json.load(f)
    if payload.get("arborenum_bounds", {}).get("objective_bound") is not None:
        return

    print(f"  [ARBORENUM] Backfilling Rashomon cutoff for {out_dir.name}...")
    X_train = pd.read_csv(interval_dir / "X_train.csv")
    y_train = pd.read_csv(interval_dir / "y_train.csv").iloc[:, 0]
    model, bounds = fit_model(X_train, y_train)

    n_trees = model.count_trees()
    if n_trees != payload["n_total_trees"]:
        print(f"  [ARBORENUM] Refit found {n_trees} trees, expected {payload['n_total_trees']}; "
              "cutoff not saved.")
        return

    payload["arborenum_bounds"] = bounds
    with open(interval_dir / "interval_inputs.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [ARBORENUM] Saved objective bound {bounds['objective_bound']}.")


def save_interval_inputs(model, X_train, y_train, sampled_tree_indices, n_trees, out_dir, bounds):
    interval_dir = out_dir / "interval_inputs"
    interval_dir.mkdir(parents=True, exist_ok=True)

    # Save exact training data and column order
    X_train.reset_index(drop=True).to_csv(interval_dir / "X_train.csv", index=False)
    y_train.reset_index(drop=True).to_csv(interval_dir / "y_train.csv", index=False)

    # Save trees using both ArborEnum path APIs
    exported_trees = []
    for idx in sampled_tree_indices:
        try:
            paths_str, leaf_preds = model.get_tree_paths_str(idx)
            # Keep original output for inspecting the library's split representation
            raw_paths, raw_preds = model.get_tree_paths(idx)

            exported_trees.append({
                "tree_index": int(idx),
                "paths": list(paths_str),
                "predictions": [int(p) for p in leaf_preds],
                "raw_paths_repr": repr(raw_paths),
                "raw_predictions_repr": repr(raw_preds),
            })
        except Exception as e:
            print(f"Interval export failed on tree {idx}: {e}")

    payload = {
        "n_total_trees": int(n_trees),
        "n_exported_trees": len(exported_trees),
        # With early stopping, n_total_trees may itself be a partial Rashomon set
        "complete_enumeration_export": len(exported_trees) == n_trees,
        "feature_names": list(X_train.columns),
        "parameters": {
            "lambda_reg": LAMBDA_REG,
            "depth_budget": DEPTH_BUDGET,
            "rashomon_mult": RASHOMON_MULT,
            "lookahead_k": LOOKAHEAD_K,
            "early_stopping": True,
            "second_rashomon_mult": SECOND_RASHOMON_MULT,
            "multiplier_step_size": 0.0025,
            "runtime_limit_seconds": RUNTIME_LIMIT_SECONDS,
            "memory_limit_mb": MEMORY_LIMIT_MB,
            "max_tree_sample": MAX_TREE_SAMPLE,
            "random_seed": RANDOM_SEED,
        },
        # Final cutoff after early-stopping extension: trees in the set have objective <= objective_bound
        "arborenum_bounds": bounds,
        "trees": exported_trees,
    }

    with open(interval_dir / "interval_inputs.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(exported_trees)} trees for interval-split analysis.")


def run_dataset(dataset_name, cfg):
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _expected_outputs = [
        "arborenum_results.txt",
        "arborenum_first_tree.txt",
        "arborenum_tree_size.json",
        "arborenum_sampled_trees.json",
        "arborenum_first_tree_paths.txt",
        "arborenum_rid.json",
        "interval_inputs/interval_inputs.json",
        "arborenum_feature_mapping.json",
    ]
    if all((out_dir / f).exists() for f in _expected_outputs):
        print(f"  [ARBORENUM] Skipping {dataset_name} — results already exist.")
        backfill_bounds(out_dir)
        return

    X_train, X_test, y_train, y_test = load_dataset(cfg)

    print(f"  [ARBORENUM] Training on {dataset_name}...")
    start = time.perf_counter()
    model, bounds = fit_model(X_train, y_train)
    duration = time.perf_counter() - start

    export_arborenum_feature_mapping(model, out_dir)
    print_split_diagnostics(model, out_dir)

    n_trees = model.count_trees()
    if n_trees <= MAX_TREE_SAMPLE:
        sampled_tree_indices = list(range(n_trees))
    else:
        sampled_tree_indices = sorted(
            random.sample(range(n_trees), MAX_TREE_SAMPLE)
        )

    tree_idx = 0   # Keep for evaluation and visualization
    test_preds = model.get_predictions(tree_idx, X_test)
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

    with open(out_dir / "arborenum_tree_size.json", "w") as f:
        json.dump(tree_size, f, indent=2)

    all_tree_paths = []

    for idx in sampled_tree_indices:
        try:
            paths_str, leaf_preds = model.get_tree_paths_str(idx)

            all_tree_paths.append({
                "tree_index": idx,
                "paths": paths_str,
                "predictions": [int(p) for p in leaf_preds],
            })

        except Exception as e:
            print(f"Failed on tree {idx}: {e}")

    metadata = {
        "n_total_trees": n_trees,
        "n_sampled_trees": len(sampled_tree_indices),
        "trees": all_tree_paths,
    }

    with open(out_dir / "arborenum_sampled_trees.json", "w") as f:
        json.dump(metadata, f, indent=2)

    save_interval_inputs(
        model=model,
        X_train=X_train,
        y_train=y_train,
        sampled_tree_indices=sampled_tree_indices,
        n_trees=n_trees,
        out_dir=out_dir,
        bounds=bounds,
    )

    with open(out_dir / "arborenum_first_tree.txt", "w", encoding="utf-8") as fh:
        try:
            paths_str, tree_preds = model.get_tree_paths_str(tree_idx)
            fh.write("\n".join(f"{p} -> {pred}" for p, pred in zip(paths_str, tree_preds)))
        except Exception as e:
            fh.write(f"Tree unavailable: {e}\n")

    feature_names = list(X_train.columns)

    print(f"  [ARBORENUM] Computing Rashomon Importance Distribution...")
    try:
        rid_start = time.perf_counter()
        rid_out = model.compute_rid(
            X_train,
            y_train,
            early_stopping=True,
            proxy_mode="hybrid",
            lambda_reg=LAMBDA_REG,
            depth_budget=DEPTH_BUDGET,
            rashomon_mult=RASHOMON_MULT,
            lookahead_k=LOOKAHEAD_K,
            max_number_thresholds_per_feature=None,
            proxy_refinement="auto",
            second_rashomon_mult=SECOND_RASHOMON_MULT,
            multiplier_step_size=0.0025,
            runtime_limit_seconds=RUNTIME_LIMIT_SECONDS,
            memory_limit_mb=MEMORY_LIMIT_MB,
            n_boot=RID_N_BOOT,
            n_scramble_evals=RID_N_SCRAMBLE_EVALS,
            seed=RID_SEED,
            memory_efficient=False,
        )
        rid_duration = time.perf_counter() - rid_start

        with open(out_dir / "arborenum_rid.json", "w") as f:
            json.dump(
                {
                    "feature_names": feature_names,
                    "mean_sub_mr": [float(v) for v in rid_out["mean_sub_mr"]],
                },
                f,
            )

        # fig, _ = model.rid_plot_mean(feature_names=feature_names, show=False)
        # fig.savefig(out_dir / "praxis_rid_mean.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)

        # fig, _ = model.rid_plot_violin(feature_names=feature_names, show=False)
        # fig.savefig(out_dir / "praxis_rid_violin.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)

        # fig, _ = model.rid_plot_cdfs(feature_names=feature_names, show=False)
        # fig.savefig(out_dir / "praxis_rid_cdfs.png", dpi=150, bbox_inches="tight")
        # plt.close(fig)

        rid_error = None
    except Exception as e:
        rid_duration = 0.0
        rid_error = str(e)


    print("Get Tree Paths:")
    try:
        paths_str, leaf_preds = model.get_tree_paths_str(tree_idx)

        with open(out_dir / "arborenum_first_tree_paths.txt", "w", encoding="utf-8") as fh:
            for path, pred in zip(paths_str, leaf_preds):
                print(f"Path: {path} -> Prediction: {pred}")
                fh.write(f"{path} -> {pred}\n")
    except Exception as e:
        print(f"Error retrieving tree paths: {e}")

    with open(out_dir / "arborenum_results.txt", "w") as f:
        f.write(f"Accuracy: {acc}")
        f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds)}")
        f.write(f"\nClassification Report:\n{classification_report(y_test, test_preds)}")
        f.write(f"\nARBORENUM completed in {duration:.2f} seconds")
        f.write(f"\nRashomon set size: {n_trees}")
        f.write(f"\nObjective bound: {bounds['objective_bound']} "
                f"(reference objective {bounds['reference_objective']}, "
                f"minimum objective {bounds['min_objective']})")
        if "error" not in tree_size:
            f.write(f"\nTree Size (tree 0): {tree_size['n_leaves']} leaves, {tree_size['n_nodes']} total nodes")
        else:
            f.write(f"\nTree Size: Error - {tree_size['error']}")
        if rid_error is None:
            f.write(f"\nRID computed in {rid_duration:.2f} seconds (see praxis_rid.json)")
        else:
            f.write(f"\nRID: Error - {rid_error}")

    print(f"  [ARBORENUM] Accuracy: {acc:.4f} | Time: {duration:.2f}s | Trees: {n_trees}")


if __name__ == "__main__":
    for dataset_name, cfg in DATASETS.items():
        print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        try:
            run_dataset(dataset_name, cfg)
        except Exception as e:
            print(f"  [ARBORENUM] {dataset_name} failed: {e}")
