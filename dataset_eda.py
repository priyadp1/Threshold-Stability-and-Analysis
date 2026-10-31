import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Dataset_Dir = Path(__file__).resolve().parent / "DatasetWork"
save_dir = Path(__file__).resolve().parent / "EDA_Results"
hist_dir = save_dir / "histograms"
hist_dir.mkdir(parents=True, exist_ok=True)
TARGET_COL = "label"
MAX_DISCRETE_UNIQUE = 10   


def feature_type(col):
    n = col.nunique()
    if n <= 1:
        return "constant"
    if n == 2:
        return "binary"
    is_integer = pd.api.types.is_integer_dtype(col) or (col.dropna() % 1 == 0).all()
    if is_integer and n <= MAX_DISCRETE_UNIQUE:
        return "discrete"
    return "continuous"


def dataset_type(counts):
    used = {t: c for t, c in counts.items() if t != "constant" and c > 0}
    if set(used) == {"binary"}:
        return "binarized"
    if set(used) == {"continuous"}:
        return "continuous only"
    return "mixed"


def plot_histogram(df, dataset, feature, bins=40):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[feature].dropna(), bins=bins, color="#2a78d6", edgecolor="white", linewidth=1)
    ax.set_title(f"{dataset}: {feature}", loc="left", fontsize=12)
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = hist_dir / f"{dataset}__{feature}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


rows = []
feature_rows = []
for file_path in sorted(Dataset_Dir.glob("*_processed.csv")):
    dataset = file_path.stem.replace("_processed", "")
    df = pd.read_csv(file_path)
    X = df.drop(columns=[TARGET_COL])

    types = {c: feature_type(X[c]) for c in X.columns}
    counts = {t: list(types.values()).count(t) for t in ["binary", "discrete", "continuous", "constant"]}
    n_unique = X.nunique()

    for c in X.columns:
        feature_rows.append({
            "dataset": dataset,
            "feature": c,
            "dtype": str(X[c].dtype),
            "type": types[c],
            "n_unique": int(n_unique[c]),
            "candidate_thresholds": max(int(n_unique[c]) - 1, 0),
        })

    continuous = [c for c in X.columns if types[c] == "continuous"]
    rows.append({
        "dataset": dataset,
        "n_rows": len(df),
        "n_features": X.shape[1],
        "n_binary": counts["binary"],
        "n_discrete": counts["discrete"],
        "n_continuous": counts["continuous"],
        "n_non_continuous": counts["binary"] + counts["discrete"],
        "n_constant": counts["constant"],
        "dataset_type": dataset_type(counts),
        "binarized": dataset_type(counts) == "binarized",
        "total_candidate_thresholds": int((n_unique - 1).clip(lower=0).sum()),
        "max_unique_in_a_feature": int(n_unique.max()),
        "positive_rate": round(df[TARGET_COL].mean(), 3),
    })
    if continuous:
        plot_histogram(X, dataset, n_unique[continuous].idxmax())

summary = pd.DataFrame(rows)
features = pd.DataFrame(feature_rows)
summary.to_csv(save_dir / "dataset_summary.csv", index=False)
features.to_csv(save_dir / "feature_summary.csv", index=False)

with open(save_dir / "dataset_summary.txt", "w") as f:
    f.write("Dataset summary table:\n")
    f.write(summary.to_string(index=False))
    f.write("\n\nDataset types:\n")
    f.write(str(summary["dataset_type"].value_counts()))
    f.write(f"\n\nFully binarized datasets ({summary['binarized'].sum()} of {len(summary)}): ")
    f.write(", ".join(summary.loc[summary["binarized"], "dataset"]))
    only_cont = summary["dataset_type"] == "continuous only"
    f.write(f"\nContinuous-only datasets ({only_cont.sum()} of {len(summary)}): ")
    f.write(", ".join(summary.loc[only_cont, "dataset"]))
    f.write("\n\nCandidate thresholds (parameter variations) per dataset:\n")
    f.write(summary[["dataset", "total_candidate_thresholds", "max_unique_in_a_feature"]]
            .sort_values("total_candidate_thresholds", ascending=False).to_string(index=False))
    f.write("\n")

print(open(save_dir / "dataset_summary.txt").read())
if len(sys.argv) == 3:
    name, feat = sys.argv[1], sys.argv[2]
    df = pd.read_csv(Dataset_Dir / f"{name}_processed.csv")
    print("Saved", plot_histogram(df, name, feat))
