from __future__ import annotations
from typing import Dict, List
from typing import Iterable

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

def dataframe_to_records(df, *, feature_cols: Iterable[str]) -> List[Dict[str, str]]:
    """将 DataFrame 转为 LIMBO 需要的 records: List[Dict[str,str]]。
    仅使用 `feature_cols` 中的列，按字符串类型编码；缺失值跳过。
    """
    records: List[Dict[str, str]] = []
    cols = list(feature_cols)
    for _, row in df.iterrows():
        rec: Dict[str, str] = {}
        for c in cols:
            if c not in df.columns:
                continue
            v = row[c]
            if v is None:
                continue
            s = str(v)
            if s == "nan":
                continue
            rec[c] = s
        records.append(rec)
    return records

def record_to_distribution(record: Dict[str, str], attr_to_id: Dict[str, int]) -> Dict[int, float]:
    prob = 1.0 / len(record)
    # 只用已知特征，忽略新特征
    return {attr_to_id[f"{k}.{v}"]: prob for k, v in record.items() if f"{k}.{v}" in attr_to_id}

def records_to_sparse(records: List[Dict[str, str]], attr_to_id: Dict[str, int], vocab_size: int):
    """批量将 records 转为 n×V 稀疏矩阵。"""
    from .dcf import DCF
    dcfs = [DCF(1.0, record_to_distribution(r, attr_to_id)) for r in records]
    return DCF.batch_to_sparse(dcfs, vocab_size)