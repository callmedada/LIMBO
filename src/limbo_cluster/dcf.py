import numpy as np
from typing import Dict, List, Set
try:
    from scipy.sparse import csr_matrix
except ImportError:  # noqa: D401
    csr_matrix = None

class DCF:
    """Distributional Cluster Feature (sparse representation)."""

    __slots__ = ("p_c", "dist", "_keys")

    def __init__(self, p_c: float, dist: Dict[int, float]):
        self.p_c = p_c
        self.dist = dist  # sparse mapping attr_id -> prob
        self._keys: Set[int] = set(dist)

    def to_sparse(self, vocab_size: int) -> "csr_matrix":
        """Return 1×V CSR row (requires SciPy)."""
        if csr_matrix is None:
            raise RuntimeError("scipy is required for sparse backend")
        if not self.dist:
            return csr_matrix((1, vocab_size))
        idx = np.fromiter(self.dist.keys(), dtype=int)
        val = np.fromiter(self.dist.values(), dtype=float)
        return csr_matrix((val, (np.zeros_like(idx), idx)), shape=(1, vocab_size))

    @staticmethod
    def batch_to_sparse(dcfs: List["DCF"], vocab_size: int):
        """批量将 DCF 列表转为 n×V 稀疏矩阵。"""
        try:
            from scipy.sparse import vstack
        except ImportError:
            raise RuntimeError("scipy is required for sparse backend")
        rows = [dcf.to_sparse(vocab_size) for dcf in dcfs]
        return vstack(rows) if rows else None

    # ---------- distance ----------
    def js(self, other: "DCF", eps: float = 1e-12) -> float:
        """Squared Jensen‑Shannon divergence."""
        keys = self._keys | other._keys
        p = np.array([self.dist.get(k, 0.0) for k in keys])
        q = np.array([other.dist.get(k, 0.0) for k in keys])
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log2((p + eps) / (m + eps)))
        kl_qm = np.sum(q * np.log2((q + eps) / (m + eps)))
        return 0.5 * (kl_pm + kl_qm)

    @staticmethod
    def js_dense_batch(X: np.ndarray, C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        批量稠密 JS 距离，X: n×V, C: k×V，返回 n×k 距离矩阵。
        """
        n, V = X.shape
        k = C.shape[0]
        X = X[:, None, :]  # n×1×V
        C = C[None, :, :]  # 1×k×V
        M = 0.5 * (X + C)  # n×k×V
        KL_XM = np.sum(X * np.log2((X + eps) / (M + eps)), axis=2)
        KL_CM = np.sum(C * np.log2((C + eps) / (M + eps)), axis=2)
        return 0.5 * (KL_XM + KL_CM)

    @staticmethod
    def js_sparse_batch(X, C, eps: float = 1e-12):
        """
        批量稀疏 JS 距离，X: n×V, C: k×V，返回 n×k 距离矩阵。
        """
        import numpy as np
        from scipy.sparse import csr_matrix
        n, V = X.shape
        k = C.shape[0]
        # 结果矩阵
        D = np.zeros((n, k), dtype=float)
        for i in range(n):
            x = X.getrow(i)
            for j in range(k):
                c = C.getrow(j)
                m = 0.5 * (x + c)
                # 只在非零索引上遍历
                x_data = x.data
                x_idx = x.indices
                c_data = c.data
                c_idx = c.indices
                m_data = m.data
                m_idx = m.indices
                # KL(x||m)
                m_x = np.zeros_like(x_data)
                for idx, v in enumerate(x_idx):
                    m_val = m[0, v] if m[0, v] > 0 else eps
                    m_x[idx] = x_data[idx] * np.log2((x_data[idx] + eps) / (m_val + eps))
                kl_xm = np.sum(m_x)
                # KL(c||m)
                m_c = np.zeros_like(c_data)
                for idx, v in enumerate(c_idx):
                    m_val = m[0, v] if m[0, v] > 0 else eps
                    m_c[idx] = c_data[idx] * np.log2((c_data[idx] + eps) / (m_val + eps))
                kl_cm = np.sum(m_c)
                D[i, j] = 0.5 * (kl_xm + kl_cm)
        return D

    # ---------- merge ----------
    def merge(self, other: "DCF") -> "DCF":
        new_p = self.p_c + other.p_c
        out: Dict[int, float] = {}
        for k, v in self.dist.items():
            out[k] = out.get(k, 0.0) + self.p_c * v
        for k, v in other.dist.items():
            out[k] = out.get(k, 0.0) + other.p_c * v
        for k in out:
            out[k] /= new_p
        return DCF(new_p, out)