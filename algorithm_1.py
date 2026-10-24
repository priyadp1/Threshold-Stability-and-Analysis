import ast
import csv
import json
import math
import re
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd


def parse_mapping(path, columns):
    raw = json.loads(Path(path).read_text())
    entry = raw['get_internal_feature_names']
    if entry.get('status') != 'success':
        raise ValueError('Internal feature names were not exported successfully')
    names = ast.literal_eval(entry['result'])
    out = []
    for i, s in enumerate(names):
        m = re.fullmatch(r'f(\d+) <= ([-+\deE.]+)', s)
        if not m:
            raise ValueError(f'Cannot parse internal feature {i}: {s}')
        j, cut = int(m.group(1)), float(m.group(2))
        if j >= len(columns):
            raise ValueError(f'Feature index {j} exceeds X_train columns')
        out.append((j, cut))
    continuous = set(ast.literal_eval(raw['continuous_feature_indices']))
    return out, continuous


def decode_tree(record, mapping, continuous):
    root = {}
    for p, label in zip(record['paths'], record['predictions']):
        tokens = re.findall(r'[+-]\d+', p)
        signed = [int(t) for t in tokens]
        if len(tokens) != len(ast.literal_eval(p.replace('+', ''))):
            raise ValueError(f'Unrecognized signed path: {p}')
        cur = root
        for token, signed_id in zip(tokens, signed):
            idx = abs(int(signed_id))
            if idx >= len(mapping):
                raise ValueError(f'Internal split ID {idx} not in mapping')
            feature, threshold = mapping[idx]
            if 'leaf' in cur:
                raise ValueError('Leaf has descendants')
            if 'feature' not in cur:
                cur.update(feature=feature, threshold=threshold, continuous=feature in continuous,
                           internal_id=idx, children={})
            if (cur['feature'], cur['threshold']) != (feature, threshold):
                raise ValueError('Conflicting tests at a tree node')
            branch = 0 if token.startswith('+') else 1
            cur = cur['children'].setdefault(branch, {})
        if cur:
            raise ValueError('Repeated or overlapping leaf path')
        cur['leaf'] = int(label)

    def freeze(node):
        if 'leaf' in node:
            return ('L', node['leaf'])
        if set(node['children']) != {0, 1}:
            raise ValueError('Tree has incomplete branching')
        return ('N', node['feature'], node['continuous'], node['threshold'],
                freeze(node['children'][0]), freeze(node['children'][1]))
    return freeze(root)


def coords(tree):
    if tree[0] == 'L':
        return ()
    return ((tree[3],) if tree[2] else ()) + coords(tree[4]) + coords(tree[5])


def canonical_signature(tree):
    if tree[0] == 'L':
        return tree
    # Fixed binary cutpoint belongs to the structure.
    return ('N', tree[1], None if tree[2] else tree[3],
            canonical_signature(tree[4]), canonical_signature(tree[5]))


def make_grid(mapping, continuous):
    grids = defaultdict(set)
    for feature, cut in mapping:
        if feature in continuous:
            grids[feature].add(cut)
    return {f: sorted(v) for f, v in grids.items()}


def structure_feature_nodes(tree):
    if tree[0] == 'L':
        return ()
    return ((tree[1],) if tree[2] else ()) + structure_feature_nodes(tree[4]) + structure_feature_nodes(tree[5])


class CertificateEngine:
    def __init__(self, sig, features, grids, X, y, gamma, theta):
        self.sig, self.features, self.grids = sig, features, grids
        self.X, self.y, self.gamma, self.theta = X, y, gamma, theta
        self.full = np.arange(len(y), dtype=np.int32)
        self._cache = {}

    def certificate(self, bounds, method):
        key = (method, tuple(bounds))
        if key in self._cache:
            return self._cache[key]
        ranges = iter(bounds)
        def assign(node):
            if node[0] == 'L':
                return ('L', node[1])
            f, fixed = node[1], node[2]
            lo, hi = next(ranges) if fixed is None else (None, None)
            return ('N', f, fixed, lo, hi, assign(node[3]), assign(node[4]))
        annotated = assign(self.sig)

        def reachable_error(node, ids):
            if not len(ids):
                return np.zeros(0, dtype=np.int8)
            if node[0] == 'L':
                return (self.y[ids] != node[1]).astype(np.int8)
            _, f, fixed, lo, hi, left, right = node
            a, b = (self.grids[f][lo], self.grids[f][hi]) if fixed is None else (fixed, fixed)
            values = self.X[ids, f]
            only_left = values <= a
            only_right = values > b
            both = ~(only_left | only_right)
            ans = np.zeros(len(ids), dtype=np.int8)
            ans[only_left] = reachable_error(left, ids[only_left])
            ans[only_right] = reachable_error(right, ids[only_right])
            if both.any():
                ans[both] = np.maximum(reachable_error(left, ids[both]), reachable_error(right, ids[both]))
            return ans

        @lru_cache(maxsize=None)
        def recurse(node_path, ids_bytes):
            node = annotated
            for step in node_path:
                node = node[5 + step]
            ids = np.frombuffer(ids_bytes, dtype=np.int32)
            if node[0] == 'L':
                return math.inf if not len(ids) else float(self.gamma + np.count_nonzero(self.y[ids] != node[1]))
            _, f, fixed, lo, hi, left, right = node
            values = self.X[ids, f]
            if fixed is not None:
                mask = values <= fixed
                return recurse(node_path + (0,), ids[mask].tobytes()) + recurse(node_path + (1,), ids[~mask].tobytes())
            a, b = self.grids[f][lo], self.grids[f][hi]
            if method == 'exact':
                worst = -math.inf
                for c in self.grids[f][lo:hi+1]:
                    mask = values <= c
                    cost = recurse(node_path + (0,), ids[mask].tobytes()) + recurse(node_path + (1,), ids[~mask].tobytes())
                    worst = max(worst, cost)
                return worst
            lmask, rmask = values <= a, values > b
            uids = ids[~(lmask | rmask)]
            base = recurse(node_path + (0,), ids[lmask].tobytes()) + recurse(node_path + (1,), ids[rmask].tobytes())
            if method == 'band':
                return base + len(uids)
            if not len(uids):
                return base
            # Profile: exact maximization over local threshold, downstream worst-case per band point.
            lerr = reachable_error(left, uids)
            rerr = reachable_error(right, uids)
            uvals = self.X[uids, f]
            worst = max(float(lerr[uvals <= c].sum() + rerr[uvals > c].sum())
                        for c in self.grids[f][lo:hi+1])
            return base + worst
        score = recurse((), self.full.tobytes())
        self._cache[key] = score
        return score


def run(args):
    source = json.loads(Path(args.trees).read_text())
    Xdf = pd.read_csv(args.X)
    ydf = pd.read_csv(args.y)
    if len(ydf.columns) != 1:
        raise ValueError('y_train.csv must have exactly one column')
    X = Xdf.to_numpy(dtype=float)
    y = ydf.iloc[:, 0].to_numpy(dtype=int)
    if len(X) != len(y) or not np.isfinite(X).all() or not set(np.unique(y)) <= {0, 1}:
        raise ValueError('Invalid X/y: require same row count, finite X and binary y')
    if list(Xdf.columns) != source['feature_names']:
        raise ValueError('Training feature order differs from interval_inputs.json')
    mapping, continuous = parse_mapping(args.mapping, Xdf.columns)
    grids = make_grid(mapping, continuous)
    gamma = int(math.floor(float(source['parameters']['lambda_reg']) * len(y) + 0.5))
    eps = float(source['parameters']['rashomon_mult'])
    records = source['trees']
    if not records:
        raise ValueError('No trees in export')
    groups = defaultdict(list)
    for rec in records:
        tree = decode_tree(rec, mapping, continuous)
        sig = canonical_signature(tree)
        feats = structure_feature_nodes(tree)
        thresholds = coords(tree)
        indices = tuple(grids[f].index(c) for f, c in zip(feats, thresholds))
        groups[sig].append({'tree_index': rec['tree_index'], 'indices': indices})
    if len({r['tree_index'] for r in records}) != len(records):
        raise ValueError('Duplicate tree indices')
    # Compute optimal objective among exported trees, not assume first is optimal.
    singleton_engines = {}
    objectives = {}
    for sig, trees in groups.items():
        feats = tuple(k for k in collect_features(sig))
        engine = CertificateEngine(sig, feats, grids, X, y, gamma, 0)
        singleton_engines[sig] = engine
        for t in trees:
            box = tuple((i, i) for i in t['indices'])
            obj = engine.certificate(box, 'exact')
            if not math.isfinite(obj):
                raise ValueError(f'Tree {t["tree_index"]} has empty leaf; check split encoding')
            objectives[t['tree_index']] = obj
    best = min(objectives.values())
    # Matches ArborEnum's rashomon_budget_: llround(reference * (1 + rashomon_mult)), kept iff obj <= bound.
    # ArborEnum's reference is its proxy objective; with a complete export it should equal the best exported tree.
    theta, theta_source = float(math.floor((1 + eps) * best + 0.5)), 'inferred'
    saved_bound = source.get('arborenum_bounds', {}).get('objective_bound')
    if args.theta is not None:
        theta, theta_source = float(args.theta), 'override'
    elif saved_bound is not None:
        theta, theta_source = float(saved_bound), 'arborenum'
    # Anytime mode budgets off a proxy root objective it never logs, so (1+eps)*best underestimates it.
    # Max exported objective <= the true cutoff, so boxes certified against it stay in the Rashomon set.
    elif getattr(args, 'theta_from_max_exported', False):
        theta, theta_source = float(max(objectives.values())), 'max_exported'
    # Exact full export required for inferring theta from the best exported tree.
    elif not source.get('complete_enumeration_export', False):
        raise ValueError('Partial enumeration and no saved ArborEnum bound: re-run the ArborEnum '
                         'script to backfill it, or set THETA_OVERRIDES')
    outside = [i for i, v in objectives.items() if v > theta + 1e-8]
    if outside:
        raise ValueError(f'{len(outside)} exported trees exceed {theta_source} cutoff {theta:.6g}. '
                         'Check Rashomon parameter convention / gamma rounding or set THETA_OVERRIDES.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for method in args.methods:
        start = time.perf_counter()
        boxes = []
        coverage = set()
        for gi, (sig, trees) in enumerate(groups.items()):
            engine = singleton_engines[sig]
            vectors = [t['indices'] for t in trees]
            k = len(vectors[0])
            projection = tuple((min(v[d] for v in vectors), max(v[d] for v in vectors)) for d in range(k))
            candidates = []
            if engine.certificate(projection, method) <= theta + 1e-8:
                candidates.append(projection)
            else:
                remaining = set(range(len(trees)))
                while remaining:
                    seed = min(remaining, key=lambda i: (objectives[trees[i]['tree_index']], trees[i]['tree_index']))
                    B = tuple((v, v) for v in vectors[seed])
                    while True:
                        options = []
                        for d, (lo, hi) in enumerate(B):
                            for direction in (-1, 1):
                                new = (lo-1, hi) if direction == -1 else (lo, hi+1)
                                if new[0] < 0 or new[1] >= len(grids[structure_feature_nodes_from_sig(sig)[d]]):
                                    continue
                                proposal = B[:d] + (new,) + B[d+1:]
                                cert = engine.certificate(proposal, method)
                                if cert <= theta + 1e-8:
                                    gained = sum(all(a <= v[j] <= b for j, (a, b) in enumerate(proposal)) for i, v in enumerate(vectors) if i in remaining)
                                    options.append((-gained, cert, d, direction, proposal))
                        if not options:
                            break
                        options.sort()
                        B = options[0][-1]
                    candidates.append(B)
                    covered = {i for i in remaining if all(a <= vectors[i][d] <= b for d, (a, b) in enumerate(B))}
                    if not covered:
                        raise AssertionError('Greedy box failed to cover its seed')
                    remaining -= covered
            for B in candidates:
                cert = engine.certificate(B, method)
                if cert > theta + 1e-8:
                    raise AssertionError('Uncertified output')
                covered_ids = [t['tree_index'] for t in trees if all(a <= t['indices'][d] <= b for d, (a, b) in enumerate(B))]
                coverage.update(covered_ids)
                # Construct node-wise intervals, preserving fixed binary cutpoints.
                tree_info = describe_tree(sig, iter(B), grids, list(Xdf.columns))
                boxes.append({'structure_id': gi, 'certificate': method, 'certificate_value': cert,
                              'threshold': theta, 'family_size': math.prod(b-a+1 for a,b in B),
                              'covered_exported_tree_indices': covered_ids, 'tree': tree_info})
        expected = {t['tree_index'] for t in records}
        if coverage != expected:
            raise AssertionError(f'Coverage incomplete: {len(expected - coverage)} trees missing')
        elapsed = time.perf_counter() - start
        summary.append({'dataset': args.dataset, 'certificate': method,
                        'n_training': len(y), 'n_features': X.shape[1],
                        'n_total_trees': source['n_total_trees'], 'n_exported_trees': len(records),
                        'n_structures': len(groups), 'n_interval_trees': len(boxes),
                        'compression_ratio': round(len(records)/len(boxes), 6),
                        'covered_trees': len(coverage), 'coverage_fraction': len(coverage)/len(records),
                        'all_boxes_certified': True, 'runtime_seconds': round(elapsed, 5),
                        'gamma': gamma, 'best_exported_objective': best, 'theta': theta,
                        'theta_source': theta_source,
                        'complete_export': source.get('complete_enumeration_export', False)})
        (out / f'interval_trees_{method}.json').write_text(json.dumps({'metadata': summary[-1], 'boxes': boxes}, indent=2))
        print(f'{method:8s} {len(boxes):4d} interval trees | {len(records)/len(boxes):.3f}x | {elapsed:.2f}s | coverage {len(coverage)}/{len(records)}')
    with (out / 'algorithm1_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (out / 'algorithm1_summary.json').write_text(json.dumps(summary, indent=2))
    print('Saved results to', out)
    return summary


def collect_features(sig):
    if sig[0] == 'L':
        return ()
    return ((sig[1],) if sig[2] is None else ()) + collect_features(sig[3]) + collect_features(sig[4])


def structure_feature_nodes_from_sig(sig):
    return collect_features(sig)


def describe_tree(sig, bounds_iter, grids, columns):
    if sig[0] == 'L':
        return {'prediction': int(sig[1])}
    _, f, fixed, left, right = sig
    if fixed is None:
        lo, hi = next(bounds_iter)
        a, b = grids[f][lo], grids[f][hi]
        node = {'feature': columns[f], 'feature_index': f, 'continuous': True,
                'threshold_range': [a, b], 'candidate_indices': [lo, hi], 'candidate_count': hi-lo+1}
    else:
        node = {'feature': columns[f], 'feature_index': f, 'continuous': False,
                'threshold_range': [fixed, fixed], 'candidate_count': 1}
    node['left'] = describe_tree(left, bounds_iter, grids, columns)
    node['right'] = describe_tree(right, bounds_iter, grids, columns)
    return node


STANDARD_RESULTS_ROOT = Path(__file__).resolve().parent / 'standard_arborenum_results'
ANYTIME_RESULTS_ROOT = Path(__file__).resolve().parent / 'anytime_arborenum_results'
METHODS = ['band', 'profile', 'exact']
# Per-root theta overrides: the two runs use different lambda/Rashomon settings, so cutoffs are not shared.
# Last field: fall back to the max exported objective when no ArborEnum bound was saved (anytime only).
RESULTS_ROOTS = {
    'standard': (STANDARD_RESULTS_ROOT, {}, False),
    'anytime': (ANYTIME_RESULTS_ROOT, {}, True),
}


def run_root(label, root, theta_overrides, theta_from_max_exported):
    if not root.is_dir():
        print(f'[{label}] skipping, {root} does not exist')
        return
    all_rows = []
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = dataset_dir.name
        inputs = dataset_dir / 'interval_inputs'
        args = SimpleNamespace(
            trees=inputs / 'interval_inputs.json',
            X=inputs / 'X_train.csv',
            y=inputs / 'y_train.csv',
            mapping=dataset_dir / 'arborenum_feature_mapping.json',
            out=dataset_dir / 'algorithm1_results',
            dataset=name,
            theta=theta_overrides.get(name),
            theta_from_max_exported=theta_from_max_exported,
            methods=METHODS,
        )
        missing = [str(f.relative_to(root)) for f in (args.trees, args.X, args.y, args.mapping) if not f.exists()]
        if missing:
            print(f'[{label}/{name}] skipping, missing: {", ".join(missing)}')
            continue
        print(f'[{label}/{name}] running Algorithm 1')
        try:
            all_rows.extend({'run': label, **row} for row in run(args))
        except Exception as e:
            print(f'[{label}/{name}] failed: {e}')
    if all_rows:
        combined = root / 'algorithm1_all_results.csv'
        with combined.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
        print('Saved combined results to', combined)


def main():
    for label, (root, theta_overrides, theta_from_max_exported) in RESULTS_ROOTS.items():
        run_root(label, root, theta_overrides, theta_from_max_exported)


if __name__ == '__main__':
    main()
