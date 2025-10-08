from __future__ import annotations
import heapq
from typing import Dict, List, Tuple, Any
import numpy as np
from .dcf import DCF
from .utils import encode_records
from concurrent.futures import ThreadPoolExecutor

class LimboAgglomerative:
    """Information‑theoretic agglomerative clustering for categorical data."""

    def __init__(self, n_clusters: int = 2, *, n_jobs: int = 1, use_sparse: bool = True, tau: float | None = None):
        self.n_clusters = n_clusters
        self.n_jobs = max(1, n_jobs)
        self.use_sparse = use_sparse
        self.tau = tau  # 距离阈值：当最小合并代价（此处用 JS 距离近似 δI）超过 tau 时停止合并
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
            if self.tau is not None and d > self.tau:
                break
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
        # linkage matrix 自动补全到 n-1 行
        # 剩余 active 中的簇依次合并，距离填0
        cur_id = len(dcfs)
        act = list(active)
        while len(linkage) < n - 1:
            i = act.pop()
            j = act.pop()
            size = cluster_sizes[i] + cluster_sizes[j]
            linkage.append([i, j, 0.0, size])
            cluster_sizes.append(size)
            act.append(cur_id)
            cur_id += 1
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

    def summary(self, *, top_k: int = 5, with_matrix: bool = False) -> Dict[str, Any]:
        """
        返回聚类摘要：每簇规模、簇内多样性（熵）、代表特征（DCF Top-K），
        以及簇间 JS 分离度统计（min/mean/max）。
        Parameters:
            top_k: 每簇返回的高权重特征数
            with_matrix: 是否返回簇间 JS 距离矩阵
        """
        if self.clusters_ is None or self.labels_ is None or self._id2attr is None:
            raise RuntimeError("You must fit first")
        k = len(self.clusters_)
        # sizes
        import numpy as np
        sizes = np.bincount(self.labels_, minlength=k).astype(int)
        # entropy per cluster (diversity)
        entropies: List[float] = []
        eps = 1e-12
        for c in self.clusters_:
            if not c.dist:
                entropies.append(0.0)
                continue
            p = np.fromiter(c.dist.values(), dtype=float)
            entropies.append(float(-(p * np.log2(p + eps)).sum()))
        # top features per cluster
        top_features: List[List[Tuple[str, float]]] = []
        for c in self.clusters_:
            items = sorted(c.dist.items(), key=lambda kv: -kv[1])[: max(0, top_k)]
            pretty = [(self._id2attr[idx], round(val, 4)) for idx, val in items]
            top_features.append(pretty)
        # inter-cluster separation by JS distance
        from .dcf import DCF
        if k >= 2:
            D = np.zeros((k, k), dtype=float)
            for i in range(k):
                for j in range(i + 1, k):
                    d = self.clusters_[i].js(self.clusters_[j])
                    D[i, j] = D[j, i] = d
            # 统计上三角非零部分（或非对角）
            mask = ~np.eye(k, dtype=bool)
            vals = D[mask]
            # 只在有意义时计算
            sep = {
                "min_js": float(vals.min()) if vals.size else 0.0,
                "mean_js": float(vals.mean()) if vals.size else 0.0,
                "max_js": float(vals.max()) if vals.size else 0.0,
            }
            dist_mat = D.tolist() if with_matrix else None
        else:
            sep = {"min_js": 0.0, "mean_js": 0.0, "max_js": 0.0}
            dist_mat = [[0.0]] if with_matrix else None
        out: Dict[str, Any] = {
            "n_clusters": k,
            "sizes": sizes.tolist(),
            "entropy": [round(x, 6) for x in entropies],
            "top_features": top_features,
            "separation": sep,
        }
        if with_matrix:
            out["distance_matrix"] = dist_mat
        return out
