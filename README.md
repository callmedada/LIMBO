# LIMBO: Information-Theoretic Agglomerative Clustering for Categorical Data

LIMBO is a Python package for Limbo Paper: 
[LIMBO: Scalable Clustering of Categorical Data](https://link.springer.com/chapter/10.1007/978-3-540-24741-8_9)

It supports both dense and sparse (scipy) vectorization, multi-threaded prediction, and robust handling of edge cases. 

## Features
- Fast agglomerative clustering for categorical data
- Supports both dense (numpy) and sparse (scipy.sparse) vectorization
- Multi-threaded prediction (`n_jobs`)
- Automatic selection of dense/sparse backend for efficiency
- Robust to missing or unseen features in prediction
- Fully tested with pytest (accuracy, edge cases, performance)

## Quick Start: Install All Dependencies

You can install all dependencies automatically using `pip` (requires pip >= 23.1):

```bash
# Clone the repository
 git clone https://github.com/callmedada/LIMBO.git
 cd LIMBO

# Install all dependencies and the package itself
pip install .

# For development (editable mode):
pip install -e .
```

## Quick Start

```python
from limbo_cluster import LimboAgglomerative

data = [
    {"color": "red", "shape": "circle"},
    {"color": "red", "shape": "circle"},
    {"color": "blue", "shape": "square"},
    {"color": "blue", "shape": "square"},
    {"color": "green", "shape": "triangle"},
    {"color": "green", "shape": "triangle"},
]

model = LimboAgglomerative(n_clusters=3, n_jobs=2, use_sparse=True)
model.fit(data)

new_records = [
    {"color": "red", "shape": "circle"},
    {"color": "blue", "shape": "square"},
    {"color": "green", "shape": "triangle"},
    {"color": "red", "shape": "triangle"},
]
preds = model.predict(new_records)
print(preds)  # Output: [0, 1, 2, ...]

profiles = model.cluster_profiles()
for i, prof in enumerate(profiles):
    print(f"Cluster {i}: {prof}")
```

## API Reference

### LimboAgglomerative

```python
LimboAgglomerative(
    n_clusters: int = 2,
    n_jobs: int = 1,
    use_sparse: bool = True
)
```
- `n_clusters`: Number of clusters to find
- `n_jobs`: Number of threads for parallel prediction
- `use_sparse`: Use sparse backend (auto-selects if None)

#### Methods
- `fit(records: List[Dict[str, str]])`: Fit the model to a list of categorical records
- `predict(records: List[Dict[str, str]], n_jobs=None, use_sparse=None) -> List[int]`: Predict cluster IDs for new records
- `cluster_profiles() -> List[Dict[str, float]]`: Get feature distributions for each cluster

### Robustness
- Empty input to `fit` or `predict` raises `ValueError`
- Unseen features in prediction are ignored (robust to schema drift)

## Testing

Run all tests (requires `pytest`):
```bash
PYTHONPATH=src pytest tests/ -v
```

## License
MIT 