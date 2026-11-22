import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

standard_visualize_trees = ["diamonds" , "adult" , "heloc"]
standard_results_root = Path(__file__).resolve().parent / 'standard_arborenum_results'
plots_root = Path(__file__).resolve().parent / 'certified_tree_plots'
certificates = ["band", "profile", "exact"]

def node_label(node):
    if "prediction" in node:
        return f"predict {node['prediction']}"
    lo, hi = node["threshold_range"]
    if not node.get("continuous", True):
        return f"{node['feature']}\n(binary)"
    if lo == hi:
        return f"{node['feature']}\n≤ {lo:g}"
    return f"{node['feature']}\n≤ t, t ∈ [{lo:g}, {hi:g}]\n({node['candidate_count']} candidates)"


def layout(node, depth=0, positions=None, next_x=None):
    if positions is None:
        positions, next_x = [], [0]
    if "prediction" in node:
        x = next_x[0]
        next_x[0] += 1
        positions.append((node, x, -depth, None, None))
        return x, positions
    left_x, _ = layout(node["left"], depth + 1, positions, next_x)
    right_x, _ = layout(node["right"], depth + 1, positions, next_x)
    x = (left_x + right_x) / 2
    positions.append((node, x, -depth, (left_x, -depth - 1), (right_x, -depth - 1)))
    return x, positions


def plot_box(box, ax):
    _, positions = layout(box["tree"])
    for node, x, y, left, right in positions:
        for child, edge_label in ((left, "yes"), (right, "no")):
            if child is not None:
                ax.plot([x, child[0]], [y, child[1]], color="gray", lw=1, zorder=1)
                ax.text((x + child[0]) / 2, (y + child[1]) / 2, edge_label,
                        fontsize=7, color="dimgray", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
        is_leaf = "prediction" in node
        color = "#cfe8cf" if is_leaf and node["prediction"] == 1 else "#f4d6d6" if is_leaf else "#dde6f5"
        ax.text(x, y, node_label(node), ha="center", va="center", fontsize=8, zorder=2,
                bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="black", lw=0.8))
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.axis("off")
    ax.set_title(f"structure {box['structure_id']} | {box['certificate']} = {box['certificate_value']:g} "
                 f"(θ = {box['threshold']:g}) | covers {box['family_size']} trees", fontsize=9)


for i in standard_visualize_trees:
    results_dir = standard_results_root / i / 'algorithm1_results'
    out_dir = plots_root / i
    out_dir.mkdir(parents=True, exist_ok=True)

    print(pd.read_csv(results_dir / 'algorithm1_summary.csv')
          [["dataset", "certificate", "n_exported_trees", "n_structures", "n_interval_trees", "coverage_fraction"]])

    for cert in certificates:
        path = results_dir / f'interval_trees_{cert}.json'
        if not path.exists():
            continue
        with open(path) as f:
            boxes = json.load(f)["boxes"]

        cert_dir = out_dir / cert
        cert_dir.mkdir(exist_ok=True)
        for b, box in enumerate(boxes):
            n_leaves = len([p for p in layout(box["tree"])[1] if p[3] is None])
            fig, ax = plt.subplots(figsize=(max(6, 2.6 * n_leaves), 4.5))
            plot_box(box, ax)
            fig.tight_layout()
            fig.savefig(cert_dir / f'box_{b:03d}.png', dpi=150)
            plt.close(fig)
        print(f"{i}/{cert}: saved {len(boxes)} certified trees to {cert_dir}")
