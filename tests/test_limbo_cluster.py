import pytest
import numpy as np
from limbo_cluster import LimboAgglomerative, DCF
from limbo_cluster.utils import encode_records, record_to_distribution, records_to_sparse

@pytest.fixture
def toy_records():
    # 3个簇，每簇2条记录
    return [
        {"color": "red", "shape": "circle"},
        {"color": "red", "shape": "circle"},
        {"color": "blue", "shape": "square"},
        {"color": "blue", "shape": "square"},
        {"color": "green", "shape": "triangle"},
        {"color": "green", "shape": "triangle"},
    ]

@pytest.mark.parametrize("n_jobs,use_sparse", [(1, False), (2, False), (1, True), (2, True), (None, None)])
def test_fit_predict_accuracy(toy_records, n_jobs, use_sparse):
    # 训练
    model = LimboAgglomerative(n_clusters=3, n_jobs=n_jobs or 1, use_sparse=use_sparse if use_sparse is not None else True)
    model.fit(toy_records)
    # 预测自身，准确率应为100%
    preds = model.predict(toy_records, n_jobs=n_jobs, use_sparse=use_sparse)
    # 由于聚类簇ID顺序不定，按簇分组后检查
    label_map = {}
    for i, rec in enumerate(toy_records):
        key = tuple(rec.items())
        label_map.setdefault(key, preds[i])
    # 同簇应有相同ID
    for i, rec in enumerate(toy_records):
        key = tuple(rec.items())
        assert preds[i] == label_map[key]
    # 不同簇ID应不同
    assert len(set(label_map.values())) == 3

@pytest.mark.parametrize("use_sparse", [False, True])
def test_predict_new_records(toy_records, use_sparse):
    model = LimboAgglomerative(n_clusters=3, use_sparse=use_sparse)
    model.fit(toy_records)
    # 新样本
    new = [
        {"color": "red", "shape": "circle"},
        {"color": "blue", "shape": "square"},
        {"color": "green", "shape": "triangle"},
        {"color": "red", "shape": "triangle"},  # 混合
    ]
    preds = model.predict(new, use_sparse=use_sparse)
    # 前3个应各归属于原簇
    assert len(set(preds[:3])) == 3

@pytest.mark.parametrize("use_sparse", [False, True])
def test_dcf_to_sparse_and_batch(toy_records, use_sparse):
    encoded, attr2id, id2attr = encode_records(toy_records)
    dcfs = [DCF(1.0, {k: 1.0/len(e) for k in e}) for e in encoded]
    vocab_size = len(attr2id)
    if use_sparse:
        try:
            from scipy.sparse import csr_matrix
        except ImportError:
            pytest.skip("scipy not installed")
        mat = DCF.batch_to_sparse(dcfs, vocab_size)
        assert mat.shape == (len(dcfs), vocab_size)
        assert mat.nnz > 0
    else:
        arr = np.zeros((len(dcfs), vocab_size))
        for i, dcf in enumerate(dcfs):
            for k, v in dcf.dist.items():
                arr[i, k] = v
        assert arr.shape == (len(dcfs), vocab_size)

@pytest.mark.parametrize("use_sparse", [False, True])
def test_utils_records_to_sparse(toy_records, use_sparse):
    encoded, attr2id, id2attr = encode_records(toy_records)
    vocab_size = len(attr2id)
    if use_sparse:
        try:
            from scipy.sparse import csr_matrix
        except ImportError:
            pytest.skip("scipy not installed")
        mat = records_to_sparse(toy_records, attr2id, vocab_size)
        assert mat.shape == (len(toy_records), vocab_size)
        assert mat.nnz > 0
    else:
        arr = np.zeros((len(toy_records), vocab_size))
        for i, rec in enumerate(toy_records):
            dist = record_to_distribution(rec, attr2id)
            for k, v in dist.items():
                arr[i, k] = v
        assert arr.shape == (len(toy_records), vocab_size)

@pytest.mark.parametrize("dense", [True, False])
def test_js_batch_consistency(toy_records, dense):
    encoded, attr2id, id2attr = encode_records(toy_records)
    dcfs = [DCF(1.0, {k: 1.0/len(e) for k in e}) for e in encoded]
    vocab_size = len(attr2id)
    # 构造矩阵
    if dense:
        arr = np.zeros((len(dcfs), vocab_size))
        for i, dcf in enumerate(dcfs):
            for k, v in dcf.dist.items():
                arr[i, k] = v
        D = DCF.js_dense_batch(arr, arr)
    else:
        try:
            from scipy.sparse import csr_matrix
        except ImportError:
            pytest.skip("scipy not installed")
        mat = DCF.batch_to_sparse(dcfs, vocab_size)
        D = DCF.js_sparse_batch(mat, mat)
    # 自己和自己距离应为0
    assert np.allclose(np.diag(D), 0, atol=1e-8)
    # 距离对称
    assert np.allclose(D, D.T, atol=1e-8)

# 复杂数据集与边界测试

def make_high_dim_sparse_records(n=100, v=1000, per=3):
    # n条记录，每条v维，随机per个非零
    rng = np.random.default_rng(42)
    records = []
    for _ in range(n):
        rec = {}
        idxs = rng.choice(v, per, replace=False)
        for i in idxs:
            rec[f"f{i}"] = str(rng.integers(0, 10))
        records.append(rec)
    return records

def make_unbalanced_records():
    # 1个大簇+1个小簇
    big = [{"a": "x", "b": "y"}] * 50
    small = [{"a": "z", "b": "w"}] * 2
    return big + small

def test_high_dim_sparse():
    records = make_high_dim_sparse_records(n=50, v=500, per=2)
    model = LimboAgglomerative(n_clusters=5)
    model.fit(records)
    preds = model.predict(records)
    assert len(preds) == len(records)
    # 检查是否分成5簇
    assert len(set(preds)) == 5

def test_large_sample():
    records = make_high_dim_sparse_records(n=500, v=50, per=2)
    model = LimboAgglomerative(n_clusters=10)
    model.fit(records)
    preds = model.predict(records)
    assert len(preds) == len(records)

def test_extreme_clusters():
    records = make_high_dim_sparse_records(n=10, v=10, per=2)
    # n_clusters=1
    model1 = LimboAgglomerative(n_clusters=1)
    model1.fit(records)
    assert len(set(model1.labels_)) == 1
    # n_clusters=n
    model2 = LimboAgglomerative(n_clusters=10)
    model2.fit(records)
    assert len(set(model2.labels_)) == 10

def test_unbalanced():
    records = make_unbalanced_records()
    model = LimboAgglomerative(n_clusters=2)
    model.fit(records)
    preds = model.predict(records)
    # 应分成2簇，且大簇ID样本数远大于小簇
    counts = np.bincount(preds)
    assert np.max(counts) > 10 * np.min(counts[counts > 0])

def test_empty_input():
    model = LimboAgglomerative(n_clusters=2)
    with pytest.raises(Exception):
        model.fit([])
    with pytest.raises(Exception):
        model.predict([])

def test_missing_feature():
    records = [{"a": "1", "b": "2"}, {"a": "1"}]
    model = LimboAgglomerative(n_clusters=2)
    model.fit(records)
    # 新样本有新特征
    new = [{"a": "1", "b": "2", "c": "3"}]
    preds = model.predict(new)
    assert len(preds) == 1

def test_duplicate_records():
    records = [{"a": "1", "b": "2"}] * 10
    model = LimboAgglomerative(n_clusters=1)
    model.fit(records)
    preds = model.predict(records)
    assert all(p == preds[0] for p in preds)

def test_extreme_n_jobs():
    records = make_high_dim_sparse_records(n=20, v=20, per=2)
    model = LimboAgglomerative(n_clusters=3, n_jobs=8)
    model.fit(records)
    preds = model.predict(records, n_jobs=8)
    assert len(preds) == len(records) 