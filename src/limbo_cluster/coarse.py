from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer

from .agglomerative import LimboAgglomerative


@dataclass(frozen=True)
class ClusterInfo:
    coarse_size: int
    unique_records: int
    limbo_clusters: int


class CoarseToLimboClusterer:
    """Two-stage clustering: MiniBatchKMeans followed by per-cluster LIMBO."""

    def __init__(
        self,
        *,
        model_names: Sequence[str],
        model_costs: Sequence[float],
        quality: np.ndarray | None = None,
        beta: float = 0.3,
        coarse_clusters: int = 150,
        coarse_batch_size: int = 2048,
        coarse_max_iter: int = 200,
        limbo_clusters: int = 12,
        limbo_tau: float | None = None,
        limbo_fit_size: int = 2000,
        prob_temp: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.model_names = list(model_names)
        self.model_costs = np.asarray(model_costs, dtype=float)
        self.quality = None if quality is None else np.asarray(quality, dtype=float)
        self.beta = float(beta)
        self.coarse_clusters = int(coarse_clusters)
        self.coarse_batch_size = int(coarse_batch_size)
        self.coarse_max_iter = int(coarse_max_iter)
        self.limbo_clusters = int(limbo_clusters)
        self.limbo_tau = limbo_tau
        self.limbo_fit_size = int(limbo_fit_size)
        self.prob_temp = float(prob_temp)
        self.random_state = int(random_state)

        self._vectorizer: DictVectorizer | None = None
        self._coarse_model: MiniBatchKMeans | None = None
        self._limbo_models: Dict[int, Tuple[LimboAgglomerative, int]] = {}

        self.labels_: np.ndarray | None = None
        self.coarse_labels_: np.ndarray | None = None
        self.cluster_model_mapping_: Dict[int, str] | None = None
        self.cluster_model_probs_: Dict[int, np.ndarray] | None = None
        self.cluster_info_: Dict[int, ClusterInfo] | None = None

    def set_quality(self, quality: np.ndarray) -> None:
        self.quality = np.asarray(quality, dtype=float)

    @staticmethod
    def _deduplicate_records(
        indices: Sequence[int],
        records: Sequence[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], Dict[int, List[int]]]:
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

    def _fit_limbo_with_cap(
        self,
        unique_records: List[Dict[str, str]],
        rng: np.random.Generator,
    ) -> Tuple[LimboAgglomerative, np.ndarray]:
        n_unique = len(unique_records)
        if n_unique == 0:
            raise ValueError("No records to cluster")
        if n_unique == 1:
            labels = np.zeros(1, dtype=int)
            dummy = LimboAgglomerative(n_clusters=1, tau=self.limbo_tau)
            dummy.labels_ = labels.tolist()
            dummy.clusters_ = []
            return dummy, labels

        fit_cap = self.limbo_fit_size
        limit = n_unique if fit_cap <= 0 else min(fit_cap, n_unique)
        fit_indices = (
            np.arange(n_unique)
            if limit == n_unique
            else np.sort(rng.choice(n_unique, limit, replace=False))
        )
        fit_records = [unique_records[i] for i in fit_indices]
        k_local = max(1, min(self.limbo_clusters, len(fit_records)))

        limbo = LimboAgglomerative(n_clusters=k_local, tau=self.limbo_tau)
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

    @staticmethod
    def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        z = (x - np.max(x)) / temperature
        exp = np.exp(z)
        return exp / np.sum(exp)

    def fit(self, records: Sequence[Dict[str, str]], *, quality: np.ndarray | None = None) -> "CoarseToLimboClusterer":
        if quality is not None:
            self.set_quality(quality)
        if self.quality is None:
            raise ValueError("Quality matrix must be provided before fitting")

        records = list(records)
        n_records = len(records)
        if n_records == 0:
            raise ValueError("records cannot be empty")
        if self.quality.shape[0] != n_records:
            raise ValueError("quality rows must match number of records")
        if self.quality.shape[1] != len(self.model_names):
            raise ValueError("quality columns must match number of models")

        self._vectorizer = DictVectorizer(sparse=True)
        X = self._vectorizer.fit_transform(records)

        self._coarse_model = MiniBatchKMeans(
            n_clusters=self.coarse_clusters,
            random_state=self.random_state,
            batch_size=self.coarse_batch_size,
            max_iter=self.coarse_max_iter,
            reassignment_ratio=0.01,
        )
        coarse_labels = self._coarse_model.fit_predict(X)
        self.coarse_labels_ = coarse_labels

        final_labels = np.empty(n_records, dtype=int)
        cluster_info: Dict[int, ClusterInfo] = {}
        limbo_models: Dict[int, Tuple[LimboAgglomerative, int]] = {}
        current_offset = 0
        rng = np.random.default_rng(self.random_state)

        buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(coarse_labels):
            buckets[int(label)].append(idx)

        for coarse_id in sorted(buckets):
            idx_list = buckets[coarse_id]
            unique_records, uid_to_indices = self._deduplicate_records(idx_list, records)
            limbo, unique_labels = self._fit_limbo_with_cap(unique_records, rng)
            label_map = np.asarray(unique_labels, dtype=int)

            local_unique = len(set(label_map.tolist()))
            for uid, orig_indices in uid_to_indices.items():
                final_cluster = current_offset + label_map[uid]
                for original_idx in orig_indices:
                    final_labels[original_idx] = final_cluster

            cluster_info[coarse_id] = ClusterInfo(
                coarse_size=len(idx_list),
                unique_records=len(unique_records),
                limbo_clusters=local_unique,
            )
            limbo_models[coarse_id] = (limbo, current_offset)
            current_offset += local_unique

        self.labels_ = final_labels
        self.cluster_info_ = cluster_info
        self._limbo_models = limbo_models

        # Compute cluster-level utilities and probabilities
        mapping: Dict[int, str] = {}
        probabilities: Dict[int, np.ndarray] = {}
        for cluster_id in np.unique(final_labels):
            mask = final_labels == cluster_id
            cluster_quality = self.quality[mask].mean(axis=0)
            utilities = cluster_quality - self.beta * self.model_costs
            best_idx = int(np.argmax(utilities))
            mapping[int(cluster_id)] = self.model_names[best_idx]
            probabilities[int(cluster_id)] = self._softmax(utilities, self.prob_temp)

        self.cluster_model_mapping_ = mapping
        self.cluster_model_probs_ = probabilities

        return self

    def predict(self, records: Sequence[Dict[str, str]]) -> np.ndarray:
        if self._vectorizer is None or self._coarse_model is None:
            raise RuntimeError("You must fit the clusterer before calling predict")
        if not records:
            raise ValueError("records cannot be empty")

        X = self._vectorizer.transform(records)
        coarse_preds = self._coarse_model.predict(X)
        final_preds = np.empty(len(records), dtype=int)

        for i, coarse_id in enumerate(coarse_preds):
            coarse_id = int(coarse_id)
            if coarse_id not in self._limbo_models:
                raise ValueError(f"Coarse cluster {coarse_id} was not seen during fit")
            limbo, offset = self._limbo_models[coarse_id]
            label = limbo.predict([records[i]])[0]
            final_preds[i] = offset + label
        return final_preds

    def cluster_summary(self) -> Counter:
        if self.labels_ is None:
            raise RuntimeError("Call fit before requesting cluster summary")
        return Counter(self.labels_.tolist())


__all__ = ["CoarseToLimboClusterer", "ClusterInfo"]


