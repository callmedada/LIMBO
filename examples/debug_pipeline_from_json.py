#!/usr/bin/env python3
"""Debug script: run LIMBO clustering with a JSON input payload."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from limbo_cluster import LimboAgglomerative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to limbo_inputs_with_spacy.json",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=15,
        help="Number of clusters (k)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Quality-cost trade-off parameter beta",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Optional JS merge distance threshold tau for early stop",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample size; 0 means use all records (beware O(n^2))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top features per cluster to print",
    )
    return parser.parse_args()


def sample_records(
    records: List[Dict[str, str]],
    quality: np.ndarray,
    sample_size: int,
    seed: int,
) -> tuple[List[Dict[str, str]], np.ndarray]:
    if sample_size <= 0 or sample_size >= len(records):
        return records, quality
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), sample_size, replace=False)
    sampled_records = [records[i] for i in idx]
    sampled_quality = quality[idx]
    return sampled_records, sampled_quality


def compute_cluster_model_mapping(
    labels: np.ndarray,
    quality: np.ndarray,
    model_names: Iterable[str],
    model_costs: np.ndarray,
    beta: float,
) -> Dict[int, str]:
    model_names = list(model_names)
    cluster_to_model: Dict[int, str] = {}
    for cluster_id in range(int(labels.max()) + 1):
        mask = labels == cluster_id
        if not mask.any():
            best_idx = int(np.argmin(model_costs))
        else:
            cluster_quality = quality[mask].mean(axis=0)
            utilities = cluster_quality - beta * model_costs
            best_idx = int(np.argmax(utilities))
        cluster_to_model[cluster_id] = model_names[best_idx]
    return cluster_to_model


def summarise_utilities(
    labels: np.ndarray,
    quality: np.ndarray,
    cluster_to_model: Dict[int, str],
    model_names: Iterable[str],
    model_costs: np.ndarray,
    beta: float,
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, List[float]] = defaultdict(list)
    model_names = list(model_names)
    for idx, cluster_id in enumerate(labels):
        model_name = cluster_to_model[cluster_id]
        model_idx = model_names.index(model_name)
        utility = float(quality[idx, model_idx] - beta * model_costs[model_idx])
        stats[model_name].append(utility)
    return {
        name: {
            "count": len(utils),
            "mean": float(np.mean(utils)),
            "min": float(np.min(utils)),
            "max": float(np.max(utils)),
        }
        for name, utils in stats.items()
    }


def main() -> None:
    args = parse_args()
    if not args.json_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.json_path}")

    with args.json_path.open() as f:
        payload = json.load(f)

    records = payload["features"]
    model_names = payload["model_names"]
    model_costs = np.asarray(payload["model_costs"], dtype=float)
    quality = np.asarray(payload["quality"], dtype=float)

    print(f"Loaded {len(records)} records across {len(model_names)} models")

    records, quality = sample_records(records, quality, args.sample, args.seed)
    if args.sample and len(records) < args.sample:
        print(f"Warning: dataset smaller than requested sample, using all {len(records)} records")
    else:
        print(f"Sample size used for clustering: {len(records)}")

    model = LimboAgglomerative(n_clusters=args.clusters, tau=args.tau)
    model.fit(records)
    labels = np.asarray(model.labels_)

    cluster_sizes = Counter(labels)
    print("Cluster size distribution:")
    for cid in sorted(cluster_sizes):
        print(f"  Cluster {cid}: {cluster_sizes[cid]}")

    cluster_to_model = compute_cluster_model_mapping(
        labels, quality, model_names, model_costs, args.beta
    )
    print("\nCluster -> recommended model:")
    for cid in sorted(cluster_to_model):
        print(f"  Cluster {cid}: {cluster_to_model[cid]}")

    util_summary = summarise_utilities(
        labels, quality, cluster_to_model, model_names, model_costs, args.beta
    )
    print("\nExpected utility stats per model:")
    for model_name, stats in util_summary.items():
        print(
            "  {name}: count={count}, mean={mean:.4f}, min={min:.4f}, max={max:.4f}".format(
                name=model_name,
                **stats,
            )
        )

    summary = model.summary(top_k=args.top_k)
    print("\nCluster summary (top-{0} features):".format(args.top_k))
    for cid, (size, top_feats) in enumerate(
        zip(summary["sizes"], summary["top_features"])
    ):
        feat_str = ", ".join(f"{feat}:{score}" for feat, score in top_feats)
        print(f"  Cluster {cid} (size={size}): {feat_str}")
    print("\nInter-cluster JS separation:", summary["separation"])


if __name__ == "__main__":
    main()


