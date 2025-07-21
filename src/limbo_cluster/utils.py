from __future__ import annotations
from typing import Dict, List

def encode_records(records: List[Dict[str, str]]):
    """Return (encoded_records, attr2id, id2attr)."""
    attr2id: Dict[str, int] = {}
    encoded = []
    for rec in records:
        out = {}
        for k, v in rec.items():
            key = f"{k}.{v}"
            idx = attr2id.setdefault(key, len(attr2id))
            out[idx] = 1  # presence flag (binary). prob later.
        encoded.append(out)
    id2attr = {i: s for s, i in attr2id.items()}
    return encoded, attr2id, id2attr

def record_to_distribution(record: Dict[str, str], attr_to_id: Dict[str, int]) -> Dict[int, float]:
    prob = 1.0 / len(record)
    # 只用已知特征，忽略新特征
    return {attr_to_id[f"{k}.{v}"]: prob for k, v in record.items() if f"{k}.{v}" in attr_to_id}

def records_to_sparse(records: List[Dict[str, str]], attr_to_id: Dict[str, int], vocab_size: int):
    """批量将 records 转为 n×V 稀疏矩阵。"""
    from .dcf import DCF
    dcfs = [DCF(1.0, record_to_distribution(r, attr_to_id)) for r in records]
    return DCF.batch_to_sparse(dcfs, vocab_size)