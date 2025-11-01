from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 运行前请确保使用源码： PYTHONPATH=src python examples/routerbench_test.py ...
from limbo_cluster import LimboAgglomerative, dataframe_to_records


def main():
    parser = argparse.ArgumentParser(description="Run LIMBO clustering on routerbench_0shot.pkl")
    parser.add_argument("pkl_path", type=str, help="Path to routerbench_0shot.pkl")
    parser.add_argument("--sample", type=int, default=1000, help="Random sample size (default: 1000)")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters k (default: 10)")
    parser.add_argument("--tau", type=float, default=None, help="Distance threshold tau to early stop (default: None)")
    parser.add_argument("--feature-cols", type=str, nargs="*", default=["eval_name"], help="Feature columns used as categorical features")
    args = parser.parse_args()

    pkl = Path(args.pkl_path)
    if not pkl.exists():
        print(f"File not found: {pkl}")
        sys.exit(1)

    df = pd.read_pickle(pkl)
    print(f"df shape= {df.shape}")

    # 构造分类特征
    records = dataframe_to_records(df, feature_cols=args.feature_cols)
    print("records example:", records[0] if records else {})

    # 随机下采样，避免 O(n^2) 过大
    n = len(records)
    rng = np.random.default_rng(0)
    if args.sample and n > args.sample:
        idx = rng.choice(n, args.sample, replace=False)
        records = [records[i] for i in idx]
        print(f"sampled {len(records)} records from {n}")
    else:
        print(f"using all {n} records")

    # 聚类
    model = LimboAgglomerative(n_clusters=args.clusters, tau=args.tau)
    model.fit(records)
    print("labels len=", len(model.labels_))
    print("unique labels=", len(set(model.labels_)))
    summ = model.summary(top_k=3)
    print("summary:", summ)


if __name__ == "__main__":
    main()



