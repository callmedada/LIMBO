from __future__ import annotations
import heapq
from typing import Dict, List, Tuple
import numpy as np
from .dcf import DCF
from .utils import encode_records
from concurrent.futures import ThreadPoolExecutor

class LimboAgglomerative:
    """Information‑theoretic agglomerative clustering for categorical data."""

    def __init__(self, n_clusters: int = 2, *, n_jobs: int = 1, use_sparse: bool = True):
        self.n_clusters = n_clusters
        self.n_jobs = max(1, n_jobs)
        self.use_sparse = use_sparse
        self._attr_to_id: Dict[str, int] | None = None
        self.labels_: List[int] | None = None
        self.clusters_: List[DCF] | None = None
        self._id2attr: Dict[int, str] | None = None
        self._linkage_matrix = None  # 新增：记录聚类树

    def fit(self, records: List[Dict[str, str]]):
        if not records:
            raise ValueError("fit input cannot be empty")
        encoded, _, id2attr = encode_records(records)
        self._id2attr = id2attr
        n = len(records)
        dcfs: List[DCF] = []
        for enc in encoded:
            prob = 1.0 / len(enc)
            dist = {idx: prob for idx in enc}
            dcfs.append(DCF(1.0 / n, dist))

        # linkage matrix 记录
        linkage = []
        cluster_sizes = [1] * n
        # priority queue of pairwise JS distances
        heap: List[Tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                d = dcfs[i].js(dcfs[j])
                heapq.heappush(heap, (d, i, j))

        active = set(range(n))
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        next_cluster_id = n
        while len(active) > self.n_clusters and heap:
            d, i, j = heapq.heappop(heap)
            i = find(i)
            j = find(j)
            if i == j:
                continue
            # merge i and j -> k
            k = len(dcfs)
            dcfs.append(dcfs[i].merge(dcfs[j]))
            parent.append(k)
            parent[i] = parent[j] = k
            active.discard(i)
            active.discard(j)
            active.add(k)
            # linkage matrix: [i, j, 距离, 新簇样本数]
            size = cluster_sizes[i] + cluster_sizes[j]
            linkage.append([i, j, d, size])
            cluster_sizes.append(size)
            # push new distances
            for a in active:
                if a == k:
                    continue
                hk, ha = (k, a) if k < a else (a, k)
                dist = dcfs[hk].js(dcfs[ha])
                heapq.heappush(heap, (dist, hk, ha))

        # assign labels
        cluster_map = {cid: idx for idx, cid in enumerate(active)}
        labels = []
        for i in range(len(records)):
            root = find(i)
            labels.append(cluster_map[root])
        self.labels_ = labels
        self.clusters_ = [dcfs[cid] for cid in active]
        # linkage matrix 只保留 n-1 行（标准格式）
        self._linkage_matrix = np.array(linkage, dtype=float) if linkage else None
        return self

    def cluster_profiles(self) -> List[Dict[str, float]]:
        assert self.clusters_ is not None and self._id2attr is not None
        profiles = []
        for c in self.clusters_:
            profile = {self._id2attr[k]: round(v, 3) for k, v in c.dist.items()}
            profiles.append(dict(sorted(profile.items(), key=lambda kv: -kv[1])))
        return profiles

    def predict(self, records: List[Dict[str, str]], *, n_jobs: int | None = None, use_sparse: bool | None = None) -> List[int]:
        """
        predict the cluster of each record
        Parameters:
            records: new records
            n_jobs: number of parallel threads, default to the same as initialization
            use_sparse: whether to use sparse distance, default to smart selection (can be forced)
        Returns:
            the cluster of each record
        """
        if not records:
            raise ValueError("predict cannot be empty")
        if self.clusters_ is None or self._id2attr is None:
            raise RuntimeError("You must fit first")
        if self._attr_to_id is None:
            # 反推 attr2id
            self._attr_to_id = {v: k for k, v in self._id2attr.items()}
        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        from .utils import record_to_distribution, records_to_sparse
        from .dcf import DCF
        import numpy as np
        try:
            from scipy.sparse import csr_matrix
        except ImportError:
            csr_matrix = None
        clusters = self.clusters_
        vocab_size = len(self._id2attr)
        n = len(records)
        k = len(clusters)
        # 智能选择：如未指定 use_sparse，自动根据数据量决定
        if use_sparse is None:
            # n*V < 1e7 时优先稠密
            use_sparse = (n * vocab_size > 1e7 or k * vocab_size > 1e7) and csr_matrix is not None
        # 批量稠密
        if not use_sparse or csr_matrix is None:
            X = np.zeros((n, vocab_size), dtype=float)
            for i, r in enumerate(records):
                dist = record_to_distribution(r, self._attr_to_id)
                for k_, v in dist.items():
                    X[i, k_] = v
            C = np.zeros((k, vocab_size), dtype=float)
            for i, c in enumerate(clusters):
                for k_, v in c.dist.items():
                    C[i, k_] = v
            D = DCF.js_dense_batch(X, C)
            return list(np.argmin(D, axis=1))
        # 批量稀疏
        X = records_to_sparse(records, self._attr_to_id, vocab_size)
        C = DCF.batch_to_sparse(clusters, vocab_size)
        D = DCF.js_sparse_batch(X, C)
        return list(np.argmin(D, axis=1))

    def get_linkage_matrix(self):
        """返回 scipy dendrogram 可用的 linkage matrix (n-1, 4)"""
        return self._linkage_matrix
