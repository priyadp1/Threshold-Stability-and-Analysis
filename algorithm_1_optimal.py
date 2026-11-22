import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from algorithm_1_greedy import (
    CertificateEngine, canonical_signature, collect_features, coords, decode_tree, describe_tree,
    make_grid, parse_mapping, structure_feature_nodes,
)

TOL = 1e-8
MAX_CANDIDATES_PER_STRUCTURE = 50_000
ILP_TIME_LIMIT_SECONDS = 600.0


def load_problem(args):
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
    if len({r['tree_index'] for r in records}) != len(records):
        raise ValueError('Duplicate tree indices')
    groups = defaultdict(list)
    for rec in records:
        tree = decode_tree(rec, mapping, continuous)
        sig = canonical_signature(tree)
        indices = tuple(grids[f].index(c) for f, c in zip(structure_feature_nodes(tree), coords(tree)))
        groups[sig].append({'tree_index': rec['tree_index'], 'indices': indices})

    engines, objectives = {}, {}
    for sig, trees in groups.items():
        engine = CertificateEngine(sig, tuple(collect_features(sig)), grids, X, y, gamma, 0)
        engines[sig] = engine
        for t in trees:
            obj = engine.certificate(tuple((i, i) for i in t['indices']), 'exact')
            if not math.isfinite(obj):
                raise ValueError(f'Tree {t["tree_index"]} has empty leaf; check split encoding')
            objectives[t['tree_index']] = obj
    best = min(objectives.values())
    theta, theta_source = float(math.floor((1 + eps) * best + 0.5)), 'inferred'
    saved_bound = source.get('arborenum_bounds', {}).get('objective_bound')
    if args.theta is not None:
        theta, theta_source = float(args.theta), 'override'
    elif saved_bound is not None:
        theta, theta_source = float(saved_bound), 'arborenum'
    elif getattr(args, 'theta_from_max_exported', False):
        theta, theta_source = float(max(objectives.values())), 'max_exported'
    elif not source.get('complete_enumeration_export', False):
        raise ValueError('Partial enumeration and no saved ArborEnum bound: re-run the ArborEnum '
                         'script to backfill it, or set THETA_OVERRIDES')
    outside = [i for i, v in objectives.items() if v > theta + TOL]
    if outside:
        raise ValueError(f'{len(outside)} exported trees exceed {theta_source} cutoff {theta:.6g}. '
                         'Check Rashomon parameter convention / gamma rounding or set THETA_OVERRIDES.')
    return SimpleNamespace(source=source, columns=list(Xdf.columns), X=X, y=y, grids=grids, gamma=gamma,
                           records=records, groups=groups, engines=engines, objectives=objectives,
                           best=best, theta=theta, theta_source=theta_source)


def covered_mask(V, box):
    if not box:
        return np.ones(len(V), dtype=bool)
    lo = np.fromiter((a for a, _ in box), dtype=np.int64, count=len(box))
    hi = np.fromiter((b for _, b in box), dtype=np.int64, count=len(box))
    return np.all((V >= lo) & (V <= hi), axis=1)


def enumerate_candidates(engine, V, method, theta, limit):
    n = len(V)
    projection = tuple((int(V[:, d].min()), int(V[:, d].max())) for d in range(V.shape[1]))
    cert = engine.certificate(projection, method)
    if cert <= theta + TOL:
        # One box covers every tree of this structure: K*_s = 1 and overlap 0, so this is optimal.
        return {projection: (np.ones(n, dtype=bool), cert)}, True

    candidates, rejected = {}, set()
    frontier = []
    for i in range(n):
        box = tuple((int(v), int(v)) for v in V[i])
        if box in candidates:
            continue
        c = engine.certificate(box, method)
        if c > theta + TOL:
            raise AssertionError('Exported tree is not certified as a singleton box')
        candidates[box] = (covered_mask(V, box), c)
        frontier.append(box)

    complete = True
    while frontier and complete:
        next_frontier = []
        for box in frontier:
            mask = candidates[box][0]
            for i in np.flatnonzero(~mask):
                new = tuple((min(a, int(v)), max(b, int(v))) for (a, b), v in zip(box, V[i]))
                if new in candidates or new in rejected:
                    continue
                c = engine.certificate(new, method)
                if c > theta + TOL:
                    rejected.add(new)  # monotone certificate: no superset of this box can pass either
                    continue
                candidates[new] = (covered_mask(V, new), c)
                next_frontier.append(new)
                if len(candidates) >= limit:
                    complete = False
                    break
            if not complete:
                break
        frontier = next_frontier
    return candidates, complete


def solve_two_stage(A, time_limit):
    N, M = A.shape
    A = csr_matrix(A.astype(np.float64))
    cover = LinearConstraint(A, lb=np.ones(N), ub=np.full(N, np.inf))
    integrality = np.ones(M)
    bounds = Bounds(np.zeros(M), np.ones(M))
    opts = {'time_limit': time_limit, 'disp': False}

    t0 = time.perf_counter()
    s1 = milp(c=np.ones(M), constraints=[cover], integrality=integrality, bounds=bounds, options=opts)
    t1 = time.perf_counter()
    if s1.x is None:
        raise RuntimeError(f'Stage 1 ILP failed: {s1.message}')
    k_star = int(round(s1.x.sum()))

    # sum_i (sum_j a_ij z_j - 1) = sum_j |cov(I_j)| z_j - N
    col_counts = np.asarray(A.sum(axis=0)).ravel()
    keep_k = LinearConstraint(np.ones((1, M)), lb=k_star, ub=k_star)
    s2 = milp(c=col_counts, constraints=[cover, keep_k], integrality=integrality, bounds=bounds,
              options={**opts, 'time_limit': max(1.0, time_limit - (t1 - t0))})
    t2 = time.perf_counter()
    x = s2.x if s2.x is not None else s1.x
    selected = np.flatnonzero(x > 0.5)
    overlap = int(round(col_counts[selected].sum())) - N
    info = {'stage1_status': s1.status, 'stage2_status': s2.status,
            'stage1_seconds': t1 - t0, 'stage2_seconds': t2 - t1,
            'proven_optimal': s1.status == 0 and s2.status == 0}
    return selected, k_star, overlap, info


def run(args):
    P = load_problem(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for method in args.methods:
        start = time.perf_counter()
        cand_seconds = stage1_seconds = stage2_seconds = 0.0
        boxes, coverage = [], set()
        n_candidates, k_star_total, overlap_total = 0, 0, 0
        enumeration_complete, ilp_optimal = True, True
        for gi, (sig, trees) in enumerate(P.groups.items()):
            engine = P.engines[sig]
            V = np.array([t['indices'] for t in trees], dtype=np.int64).reshape(len(trees), -1)

            t0 = time.perf_counter()
            cands, complete = enumerate_candidates(engine, V, method, P.theta, args.max_candidates)
            cand_seconds += time.perf_counter() - t0
            enumeration_complete &= complete
            n_candidates += len(cands)

            cand_boxes = list(cands)
            if len(cand_boxes) == 1:
                selected, k_star, overlap = [0], 1, 0
            else:
                A = np.column_stack([cands[b][0] for b in cand_boxes])
                selected, k_star, overlap, info = solve_two_stage(A, args.time_limit)
                stage1_seconds += info['stage1_seconds']
                stage2_seconds += info['stage2_seconds']
                ilp_optimal &= info['proven_optimal']
            k_star_total += k_star
            overlap_total += overlap

            for j in selected:
                B = cand_boxes[j]
                mask, cert = cands[B]
                if engine.certificate(B, method) > P.theta + TOL:
                    raise AssertionError('Uncertified output')
                covered_ids = [trees[i]['tree_index'] for i in np.flatnonzero(mask)]
                coverage.update(covered_ids)
                boxes.append({'structure_id': gi, 'certificate': method, 'certificate_value': cert,
                              'threshold': P.theta, 'family_size': math.prod(b - a + 1 for a, b in B),
                              'covered_exported_tree_indices': covered_ids,
                              'tree': describe_tree(sig, iter(B), P.grids, P.columns)})
        expected = {t['tree_index'] for t in P.records}
        if coverage != expected:
            raise AssertionError(f'Coverage incomplete: {len(expected - coverage)} trees missing')
        elapsed = time.perf_counter() - start
        n_records = len(P.records)
        summary.append({'dataset': args.dataset, 'certificate': method, 'solver': 'two_stage_ilp',
                        'n_training': len(P.y), 'n_features': P.X.shape[1],
                        'n_total_trees': P.source['n_total_trees'], 'n_exported_trees': n_records,
                        'n_structures': len(P.groups), 'n_candidate_intervals': n_candidates,
                        'n_interval_trees': len(boxes), 'k_star': k_star_total,
                        'redundant_coverage': overlap_total,
                        'compression_ratio': round(n_records / len(boxes), 6),
                        'covered_trees': len(coverage), 'coverage_fraction': len(coverage) / n_records,
                        'all_boxes_certified': True,
                        'candidate_enumeration_complete': enumeration_complete,
                        'ilp_proven_optimal': ilp_optimal,
                        'proven_optimal': enumeration_complete and ilp_optimal,
                        'candidate_seconds': round(cand_seconds, 5),
                        'stage1_seconds': round(stage1_seconds, 5),
                        'stage2_seconds': round(stage2_seconds, 5),
                        'runtime_seconds': round(elapsed, 5),
                        'gamma': P.gamma, 'best_exported_objective': P.best, 'theta': P.theta,
                        'theta_source': P.theta_source,
                        'complete_export': P.source.get('complete_enumeration_export', False)})
        (out / f'interval_trees_optimal_{method}.json').write_text(
            json.dumps({'metadata': summary[-1], 'boxes': boxes}, indent=2))
        flag = '' if summary[-1]['proven_optimal'] else ' [NOT proven optimal]'
        print(f'{method:8s} {len(boxes):4d} interval trees | {n_records / len(boxes):.3f}x | '
              f'overlap {overlap_total} | {n_candidates} candidates | '
              f'time {elapsed:.2f}s (candidates {cand_seconds:.2f}s, stage1 {stage1_seconds:.2f}s, '
              f'stage2 {stage2_seconds:.2f}s) | coverage {len(coverage)}/{n_records}{flag}')
    with (out / 'algorithm1_optimal_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (out / 'algorithm1_optimal_summary.json').write_text(json.dumps(summary, indent=2))
    print('Saved results to', out)
    return summary


STANDARD_RESULTS_ROOT = Path(__file__).resolve().parent / 'standard_arborenum_results'
ANYTIME_RESULTS_ROOT = Path(__file__).resolve().parent / 'anytime_arborenum_results'
PRAIXS_RESULTS_ROOT = Path(__file__).resolve().parent / 'praxis_results'
METHODS = ['band', 'profile', 'exact']
RESULTS_ROOTS = {
    'standard': (STANDARD_RESULTS_ROOT, {}, False),
    'anytime': (ANYTIME_RESULTS_ROOT, {}, True),
    'praxis': (PRAIXS_RESULTS_ROOT, {}, False),
}


def run_root(label, root, theta_overrides, theta_from_max_exported):
    if not root.is_dir():
        print(f'[{label}] skipping, {root} does not exist')
        return
    all_rows = []
    root_start = time.perf_counter()
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
            max_candidates=MAX_CANDIDATES_PER_STRUCTURE,
            time_limit=ILP_TIME_LIMIT_SECONDS,
        )
        missing = [str(f.relative_to(root)) for f in (args.trees, args.X, args.y, args.mapping) if not f.exists()]
        if missing:
            print(f'[{label}/{name}] skipping, missing: {", ".join(missing)}')
            continue
        print(f'[{label}/{name}] running Algorithm 1 (optimal)')
        t0 = time.perf_counter()
        try:
            all_rows.extend({'run': label, **row} for row in run(args))
            print(f'[{label}/{name}] done in {time.perf_counter() - t0:.2f}s')
        except Exception as e:
            print(f'[{label}/{name}] failed after {time.perf_counter() - t0:.2f}s: {e}')
    if all_rows:
        combined = root / 'algorithm1_optimal_all_results.csv'
        with combined.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
        print('Saved combined results to', combined)
    print(f'[{label}] total time {time.perf_counter() - root_start:.2f}s')


def main():
    start = time.perf_counter()
    for label, (root, theta_overrides, theta_from_max_exported) in RESULTS_ROOTS.items():
        run_root(label, root, theta_overrides, theta_from_max_exported)
    print(f'Algorithm 1 (optimal) total time {time.perf_counter() - start:.2f}s')


if __name__ == '__main__':
    main()
