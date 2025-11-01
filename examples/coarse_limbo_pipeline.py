#!/usr/bin/env python3
"""Two-stage clustering pipeline: MiniBatchKMeans + per-cluster LIMBO."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer

from limbo_cluster import LimboAgglomerative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path, help="Path to limbo_inputs_with_spacy.json")
    parser.add_argument("--coarse-clusters", type=int, default=150, help="Number of MiniBatchKMeans clusters")
    parser.add_argument("--coarse-batch-size", type=int, default=2048, help="Mini-batch size")
    parser.add_argument("--coarse-max-iter", type=int, default=200, help="Maximum iterations for MiniBatchKMeans")
    parser.add_argument("--limbo-clusters", type=int, default=12, help="Maximum LIMBO clusters per coarse bucket")
    parser.add_argument(
        "--limbo-tau",
        type=float,
        default=None,
        help="Optional JS distance threshold tau for early stopping inside LIMBO",
    )
    parser.add_argument(
        "--limbo-fit-size",
        type=int,
        default=2000,
        help="Maximum unique records used to fit LIMBO per coarse cluster (0 means use all)",
    )
    parser.add_argument("--beta", type=float, default=0.3, help="Quality-cost trade-off coefficient")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top features to display per final cluster")
    parser.add_argument(
        "--prob-temp",
        type=float,
        default=1.0,
        help="Temperature for softmax over utilities (soft probabilities output)",
    )
    return parser.parse_args()


def load_payload(path: Path) -> Tuple[List[Dict[str, str]], np.ndarray, List[str], np.ndarray]:
    with path.open() as f:
        data = json.load(f)
    features: List[Dict[str, str]] = data["features"]
    model_names: List[str] = data["model_names"]
    model_costs = np.asarray(data["model_costs"], dtype=float)
    quality = np.asarray(data["quality"], dtype=float)
    return features, quality, model_names, model_costs


def vectorize_features(features: Sequence[Dict[str, str]]) -> Tuple[np.ndarray, DictVectorizer]:
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(features)
    return X, vec


def coarse_clustering(
    X,
    n_clusters: int,
    batch_size: int,
    max_iter: int,
    random_state: int,
) -> np.ndarray:
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(X)
    return labels


def deduplicate_records(indices: Sequence[int], records: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[int, List[int]]]:
    key_to_uid: Dict[Tuple[Tuple[str, str], ...], int] = {}
    uid_to_indices: Dict[int, List[int]] = defaultdict(list)
    unique_records: List[Dict[str, str]] = []
    for idx in indices:
        key = tuple(sorted(records[idx].items()))
        uid = key_to_uid.get(key)
        if uid is None:
            uid = len(unique_records)
            key_to_uid[key] = uid
            unique_records.append(records[idx])
        uid_to_indices[uid].append(idx)
    return unique_records, uid_to_indices


def fit_limbo_with_cap(
    unique_records: List[Dict[str, str]],
    limbo_clusters: int,
    tau: float | None,
    fit_cap: int,
    rng: np.random.Generator,
) -> Tuple[LimboAgglomerative, np.ndarray]:
    n_unique = len(unique_records)
    if n_unique == 0:
        raise ValueError("No records to cluster")
    if n_unique == 1:
        labels = np.zeros(1, dtype=int)
        dummy_model = LimboAgglomerative(n_clusters=1, tau=tau)
        dummy_model.labels_ = labels.tolist()
        dummy_model.clusters_ = []
        return dummy_model, labels

    limit = n_unique if fit_cap <= 0 else min(fit_cap, n_unique)
    fit_indices = np.arange(n_unique) if limit == n_unique else np.sort(rng.choice(n_unique, limit, replace=False))
    fit_records = [unique_records[i] for i in fit_indices]
    k_local = min(limbo_clusters, len(fit_records))
    k_local = max(1, k_local)

    limbo = LimboAgglomerative(n_clusters=k_local, tau=tau)
    limbo.fit(fit_records)
    fit_labels = np.asarray(limbo.labels_, dtype=int)

    all_labels = np.empty(n_unique, dtype=int)
    all_labels[fit_indices] = fit_labels
    remaining = np.setdiff1d(np.arange(n_unique), fit_indices, assume_unique=True)
    if remaining.size:
        remaining_records = [unique_records[i] for i in remaining]
        pred_labels = np.asarray(limbo.predict(remaining_records), dtype=int)
        all_labels[remaining] = pred_labels

    return limbo, all_labels


def assign_final_labels(
    records: Sequence[Dict[str, str]],
    coarse_labels: np.ndarray,
    limbo_clusters: int,
    tau: float | None,
    fit_cap: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[int, Dict[str, int]]]:
    n = len(records)
    final_labels = np.empty(n, dtype=int)
    cluster_info: Dict[int, Dict[str, int]] = {}
    current_offset = 0

    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(coarse_labels):
        buckets[int(label)].append(idx)

    for coarse_id in sorted(buckets):
        idx_list = buckets[coarse_id]
        unique_records, uid_to_indices = deduplicate_records(idx_list, records)

        limbo, unique_labels = fit_limbo_with_cap(unique_records, limbo_clusters, tau, fit_cap, rng)
        label_map = np.asarray(unique_labels, dtype=int)
        local_unique = len(set(label_map.tolist()))

        local_counts = Counter(label_map.tolist())

        for uid, orig_indices in uid_to_indices.items():
            final_cluster = current_offset + label_map[uid]
            for orig_idx in orig_indices:
                final_labels[orig_idx] = final_cluster

        cluster_info[coarse_id] = {
            "coarse_size": len(idx_list),
            "unique_records": len(unique_records),
            "limbo_clusters": local_unique,
        }

        current_offset += local_unique

    return final_labels, cluster_info


def compute_cluster_model_mapping(
    labels: np.ndarray,
    quality: np.ndarray,
    model_names: Sequence[str],
    model_costs: np.ndarray,
    beta: float,
) -> Tuple[Dict[int, str], Dict[int, np.ndarray]]:
    mapping: Dict[int, str] = {}
    probabilities: Dict[int, np.ndarray] = {}
    unique_clusters = np.unique(labels)
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        cluster_quality = quality[mask].mean(axis=0)
        utilities = cluster_quality - beta * model_costs
        best_model_idx = int(np.argmax(utilities))
        mapping[int(cluster_id)] = model_names[best_model_idx]
        probabilities[int(cluster_id)] = utilities
    return mapping, probabilities


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = (x - np.max(x)) / temperature
    exp = np.exp(z)
    return exp / np.sum(exp)


def summarise_models(
    labels: np.ndarray,
    quality: np.ndarray,
    cluster_model_mapping: Dict[int, str],
    model_names: Sequence[str],
    model_costs: np.ndarray,
    beta: float,
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, List[float]] = defaultdict(list)
    model_names = list(model_names)
    for idx, cluster_id in enumerate(labels):
        model_name = cluster_model_mapping[int(cluster_id)]
        model_idx = model_names.index(model_name)
        utility = float(quality[idx, model_idx] - beta * model_costs[model_idx])
        stats[model_name].append(utility)
    return {
        name: {
            "count": len(values),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in stats.items()
    }


def cluster_summary(labels: np.ndarray) -> Counter:
    return Counter(labels.tolist())


def main() -> None:
    args = parse_args()
    if not args.json_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.json_path}")

    records, quality, model_names, model_costs = load_payload(args.json_path)
    print(f"Loaded {len(records)} records across {len(model_names)} models")

    X, _ = vectorize_features(records)
    coarse_labels = coarse_clustering(
        X,
        n_clusters=args.coarse_clusters,
        batch_size=args.coarse_batch_size,
        max_iter=args.coarse_max_iter,
        random_state=args.random_state,
    )
    coarse_counts = Counter(coarse_labels.tolist())
    print("Coarse cluster distribution (top 10):")
    for cid, cnt in coarse_counts.most_common(10):
        print(f"  Cluster {cid}: {cnt}")

    rng = np.random.default_rng(args.random_state)
    final_labels, cluster_info = assign_final_labels(
        records,
        coarse_labels,
        limbo_clusters=args.limbo_clusters,
        tau=args.limbo_tau,
        fit_cap=args.limbo_fit_size,
        rng=rng,
    )

    print("Per coarse cluster stats:")
    for cid in sorted(cluster_info):
        info = cluster_info[cid]
        print(
            f"  Coarse {cid}: size={info['coarse_size']}, unique={info['unique_records']}, limbo_clusters={info['limbo_clusters']}"
        )

    cluster_counts = cluster_summary(final_labels)
    print(f"Total fine clusters: {len(cluster_counts)}")
    print("Fine cluster size distribution (top 10):")
    for cid, cnt in cluster_counts.most_common(10):
        print(f"  Cluster {cid}: {cnt}")

    cluster_model_mapping, raw_utilities = compute_cluster_model_mapping(
        final_labels, quality, model_names, model_costs, args.beta
    )
    cluster_probabilities = {
        cid: softmax(util, temperature=args.prob_temp)
        for cid, util in raw_utilities.items()
    }
    model_stats = summarise_models(
        final_labels, quality, cluster_model_mapping, model_names, model_costs, args.beta
    )

    print("Cluster -> recommended model:")
    for cid in sorted(cluster_model_mapping):
        probs = cluster_probabilities[cid]
        prob_str = ", ".join(
            f"{model_names[i]}:{prob:.3f}" for i, prob in enumerate(probs)
        )
        print(f"  Cluster {cid}: {cluster_model_mapping[cid]} | probs=[{prob_str}]")

    print("\nModel utility stats:")
    for model_name, stats in model_stats.items():
        print(
            "  {name}: count={count}, mean={mean:.4f}, min={min:.4f}, max={max:.4f}".format(
                name=model_name,
                **stats,
            )
        )


if __name__ == "__main__":
    main()


