#!/usr/bin/env python
# coding: utf-8

# #LIMBO Section

# #### **Assumption**
# 
# Assume that we have the NLP extracted feature values for LIMBO to cluster for all the queries in our training corpus ($Q$) and they are transformed into a new DataFrame $Q_{\text{LIMBO}}$ with a defined transforming function $F$. Assume a feature extraction function $F: Q \to Q_{\text{LIMBO}}$ that maps each query to its corresponding $f$-dimensional feature vector.
# 
# #### **Notation**
# 
# Let:
# 
# - $Q_{\text{LIMBO}} = \{q_1, q_2, \dots, q_n\}$ be the **feature values corpus of queries** for training LIMBO.
# - Each query $q \in Q_{\text{LIMBO}}$ is represented by an $f$-dimensional feature vector:
# 
#   $$
#   \text{Features: } [\text{semantic complexity},\ \text{technical complexity},\ \text{related field},\ \text{factuality}]
#   $$
# 
# - $M = \{M_1, M_2, \dots, M_m\}$ is the set of **candidate LLMs**.
# 
# ---
# 
# #### **Model Training**
# 
# 1. The query feature vectors are clustered into $k$ clusters:
# 
#    $$
#    C = \{c_1, c_2, \dots, c_k\}
#    $$
# 
# 2. For each cluster $c_i$, assign the optimal model $M_L \in M$ that **maximizes expected utility (normalization needed potentially)**:
# 
#    $$
#    U(Q_i, M) = \beta \cdot \text{Quality}(Q_i, M) - (1 - \beta) \cdot \text{Cost}(M)
#    $$
# 
#    - $Q_i \subseteq Q_{\text{LIMBO}}$: the queries in cluster $c_i$
#    - $\text{Quality}(Q_i, M)$: empirical model performance on $Q_i$
#    - $\text{Cost}(M)$: normalized cost of using model $M$
#    - $\beta \in [0, 1]$: user-defined trade-off parameter between quality and cost
# 
# ---
# 
# #### **Model Inference**
# 
# Given a new query $q_{\text{new}}$:
# 
# 1. Extract feature vector $\vec{f}_{\text{new}} \in \mathbb{R}^f$
# 2. Assign $q_{\text{new}}$ to a cluster $c_j$
# 3. Select the model that maximizes utility:
# 
# $$
# M_{\text{new}} = M_j \quad \text{where } q_{\text{new}} \in c_j \text{ and } M_j = \arg\max_{M_L \in M} \ U(Q_j, M_L)
# $$
# 
# 
# 
# ---
# 
# #### **Next Steps**
# 
# Maybe create a **self-supervised loop** where cluster-to-model mappings are updated over time with new quality score, user feedback, etc.
# 

# In[33]:


import numpy as np
from typing import List, Dict, Tuple, Optional
from limbo_cluster import LimboAgglomerative


# In[37]:


def train_LIMBO(Q: List[Dict[str, str]], M: List[str], Cost: List[float], 
                k: int, beta: float, Quality: Optional[np.ndarray] = None) -> Tuple[List[int], List[str], Dict[int, str]]:
    limbo = LimboAgglomerative(n_clusters=k)
    limbo.fit(Q)
    clusters = limbo.labels_
    
    # edge case, no quality data provided
    if Quality is None:
        n_queries = len(Q)
        n_models = len(M)
        Quality = np.random.rand(n_queries, n_models)  # mock up fir the test purposes #TODO: change it
    
    model_of_the_cluster = []
    cluster_model_mapping = {}
    
    for cluster_id in range(k):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        
        if not cluster_indices:
            best_model_idx = np.argmin(Cost)
            model_of_the_cluster.append(M[best_model_idx])
            cluster_model_mapping[cluster_id] = M[best_model_idx]
            continue
        
        cluster_quality = np.mean(Quality[cluster_indices], axis=0)
        
        utilities = cluster_quality - beta * np.array(Cost)
        
        best_model_idx = np.argmax(utilities)
        model_of_the_cluster.append(M[best_model_idx])
        cluster_model_mapping[cluster_id] = M[best_model_idx]
    
    return clusters, model_of_the_cluster, cluster_model_mapping

def run_LIMBO(query: Dict[str, str], cluster_model_mapping: Dict[int, str], 
              limbo_model: LimboAgglomerative, M: List[str], Cost: List[float], 
              beta: float, Quality_new: Optional[np.ndarray] = None) -> Tuple[int, float, str]:
    
    cluster_labels = limbo_model.predict([query])
    cluster = cluster_labels[0]
    
    recommended_model = cluster_model_mapping[cluster]
    model_idx = M.index(recommended_model)
    
    if Quality_new is None:
        n_models = len(M)
        quality_new = np.random.rand(n_models)  # 临时使用随机质量,例子
    else:
        quality_new = Quality_new
    
    expected_utility = quality_new[model_idx] - beta * Cost[model_idx]
    
    return cluster, expected_utility, recommended_model

def run_LIMBO_batch(queries: List[Dict[str, str]], cluster_model_mapping: Dict[int, str], 
                    limbo_model: LimboAgglomerative, M: List[str], Cost: List[float], 
                    beta: float, Quality_new: Optional[np.ndarray] = None) -> Tuple[List[int], List[float], List[str]]:
    cluster_labels = limbo_model.predict(queries)
    expected_utilities = []
    recommended_models = []
    
    for i, cluster in enumerate(cluster_labels):
        recommended_model = cluster_model_mapping[cluster]
        model_idx = M.index(recommended_model)
        
        if Quality_new is None:
            n_models = len(M)
            quality_new = np.random.rand(n_models)
        else:
            quality_new = Quality_new[i] if Quality_new.ndim > 1 else Quality_new
        
        expected_utility = quality_new[model_idx] - beta * Cost[model_idx]
        expected_utilities.append(expected_utility)
        recommended_models.append(recommended_model)
    
    return cluster_labels, expected_utilities, recommended_models

def example_usage():

    Q = [
        {"sentiment": "positive", "factual": "high", "abc123": "low"},
        {"sentiment": "negative", "factual": "medium", "abc123": "high"},
        {"sentiment": "neutral", "factual": "high", "abc123": "medium"},
        {"sentiment": "positive", "factual": "low", "abc123": "low"},
        {"sentiment": "negative", "factual": "high", "abc123": "high"},
    ]
    
    M = ["gpt-3.5-turbo", "gpt-4", "claude-3", "llama-2"]
    Cost = [0.001, 0.03, 0.015, 0.002] 
    k = 2
    beta = 0.1  # 系数假设需要
    
    # 质量数据
    Quality = np.array([
        [0.8, 0.9, 0.85, 0.7], 
        [0.6, 0.8, 0.75, 0.6],
        [0.7, 0.85, 0.8, 0.65],
        [0.9, 0.95, 0.9, 0.8],
        [0.5, 0.7, 0.65, 0.5],
    ])
    
    # fit LIMBO
    clusters, model_of_cluster, cluster_model_mapping = train_LIMBO(Q, M, Cost, k, beta, Quality)
    
    print("res:", clusters)
    print("recommand model:", model_of_cluster)
    print("cluster-recommandation:", cluster_model_mapping)
    
    limbo_model = LimboAgglomerative(n_clusters=k)
    limbo_model.fit(Q)
    
    new_query = {"sentiment": "positive", "factual": "medium", "abc123": "low"}
    cluster, utility, model = run_LIMBO(new_query, cluster_model_mapping, limbo_model, M, Cost, beta)
    
    print(f"new query: {cluster}")
    print(f"utility: {utility:.4f}")
    print(f"recommandation: {model}")

example_usage()


# In[ ]:


# LIMBO training function needed
# LIMBO training function that takes the LIMBO features of corpus Q, list of LLMs M, a list of cost of the LLMs Cost, number of clusters k, quality-cost trade-off coefficient beta, and an array quality of the queries on each model Quality.
# training LIMBO should give a cluster --> model mapping for each cluster
# I am not sure if we have the Quality data before hand.
def train_LIMBO(Q, M, Cost, k, beta, Quality):
  # I do not know how to write this
  return cluster, model_of_the_cluster, cluster_model_mapping # cluster and model_of_cluster should be lists with k length. cluster_model_mapping is an array with all the clusters and their respective model with the highest U.


# In[ ]:


# LIMBO helper function needed here
# LIMBO helper function should take a new query or a list of queries as input (the mapping too for decision making).
# It will provide the cluster membership, expected utility of this new query, and its decision of the recommended model based on the mapping generated.
# If we update the database, we need to run the query on all the models again, which cannot help us save the cost.
# Maybe we can do that when a new query is significantly different from the existing clusters.
def run_LIMBO(query, cluster_model_mapping):
  # I do not know how to write this
  return cluster, expected_utility, recommended_model


# # BERT Section

# #### **Assumption**
# 
# Assume that we use a sentence-level BERT encoder to convert all queries in the training corpus ($Q$) into fixed-length embeddings. These embeddings are transformed into a new representation $Q_{\text{BERT}}$ via an encoding function $E$:
# 
# $$
# E: Q \to Q_{\text{BERT}} \subseteq \mathbb{R}^d
# $$
# 
# where each $q \in Q$ is mapped to a dense $d$-dimensional embedding.
# 
# #### **Notation**
# 
# Let:
# 
# - $Q_{\text{BERT}} = \{e_1, e_2, \dots, e_n\}$ be the **BERT-encoded corpus of queries**, where $e_i \in \mathbb{R}^d$.
# - $M = \{M_1, M_2, \dots, M_m\}$ is the set of **candidate LLMs**.
# 
# ---
# 
# #### **Model Training**
# 
# 1. The encoded vectors in $Q_{\text{BERT}}$ are clustered into $k$ clusters based on **cosine similarity**:
# 
#    $$
#    C = \{c_1, c_2, \dots, c_k\}
#    $$
# 
# 2. For each cluster $c_i$, assign the optimal model $M_L \in M$ that **maximizes expected utility**:
# 
#    $$
#    U(Q_i, M) = \beta \cdot \text{Quality}(Q_i, M) - (1 - \beta) \cdot \text{Cost}(M)
#    $$
# 
#    - $Q_i \subseteq Q_{\text{BERT}}$: queries in cluster $c_i$
#    - $\text{Quality}(Q_i, M)$: empirical model performance on $Q_i$
#    - $\text{Cost}(M)$: normalized cost of using model $M$
#    - $\beta \in [0, 1]$: user-defined trade-off parameter
# 
# ---
# 
# #### **Model Inference**
# 
# Given a new query $q_{\text{new}}$:
# 
# 1. Encode it via BERT to obtain a vector $\vec{e}_{\text{new}} \in \mathbb{R}^d$
# 2. Assign it to the closest cluster $c_j$ based on cosine similarity
# 3. Select the model assigned to that cluster during training:
# 
#    $$
#    M_{\text{new}} = M_j \quad \text{where } \vec{e}_{\text{new}} \in c_j \text{ and } M_j = \arg\max_{M_L \in M} \ U(Q_j, M_L)
#    $$
# 
# ---
# 
# #### **Next Steps**
# 
# Extend the system with a **self-supervised loop** that updates cluster-to-model assignments over time like what we want to do with LIMBO.
# 

# In[ ]:


# BERT training function needed
# BERT training function that takes the BERT transformed corpus Q, list of LLMs M, a list of cost of the LLMs Cost, number of clusters k, quality-cost trade-off coefficient beta, and an array quality of the queries on each model Quality.
# training BERT should give a cluster --> model mapping for each cluster
# I am not sure if we have the Quality data before hand.
def train_BERT(Q, M, Cost, k, beta, Quality):
  # BERT training function
  return cluster, model_of_the_cluster, cluster_model_mapping # cluster and model_of_cluster should be lists with k length. cluster_model_mapping is an array with all the clusters and their respective model with the highest U.


# In[ ]:


# BERT helper function needed here
# BERT helper function should take a new query or a list of queries as input (the mapping too for decision making).
# It will provide the cluster membership, expected utility of this new query, and its decision of the recommended model based on the mapping generated.
# If we update the database, we need to run the query on all the models again, which cannot help us save the cost.
# Maybe we can do that when a new query is significantly different from the existing clusters.
def run_BERT(query, cluster_model_mapping):
  # BERT inference function
  return cluster, expected_utility, recommended_model


# # Self-Supervised MLP Model

# #### **Assumption**
# 
# Assume that we have two representations of the query corpus $Q$ (mentioned in the previous sections):
# 
# - $Q_{\text{LIMBO}}$ is the feature-based representation of each query (e.g., semantic complexity, technical complexity, etc.), produced by a transformation function $F: Q \to Q_{\text{LIMBO}}$.
# - $Q_{\text{BERT}}$ is the dense embedding of each query, produced by a BERT encoder $E: Q \to Q_{\text{BERT}} \subseteq \mathbb{R}^d$.
# 
# These two representations are normalized separately via min-max scaling to ensure comparability and prevent the MLP from over-relying on the higher-dimensional BERT features.
# 
# ---
# 
# #### **Notation**
# 
# Let:
# 
# - $\hat{y}_{\text{LIMBO}}$, $\hat{y}_{\text{BERT}} \in \mathbb{R}^m$ be the soft label predictions from LIMBO and BERT, respectively, for $m$ candidate LLMs.
# - $\lambda \in [0, 1]$ be a user-defined hyperparameter that determines the weight of LIMBO vs. BERT in the soft label. (we can also make this learned from doing a value sweeping, mentioned in Next Steps).
# 
# Then the combined soft label used as supervision for the MLP is:
# 
# $$
# \hat{y}_{\text{target}} = \lambda \cdot \hat{y}_{\text{LIMBO}} + (1 - \lambda) \cdot \hat{y}_{\text{BERT}}
# $$
# 
# ---
# 
# #### **Model Training**
# 
# 1. Concatenate the normalized LIMBO and BERT representations:
# 
#    $$
#    \vec{x}_i = [\text{Normalize}(q^{\text{LIMBO}}_i); \ \text{Normalize}(q^{\text{BERT}}_i)] \in \mathbb{R}^{f + d}
#    $$
# 
# 2. Train a multi-layer perceptron (MLP) $f_\theta: \mathbb{R}^{f+d} \to \mathbb{R}^m$ to predict the soft label $\hat{y}_{\text{target}}$ using cross-entropy loss:
# 
#    $$
#    \mathcal{L} = \text{CrossEntropy}(f_\theta(\vec{x}_i), \hat{y}_{\text{target},i})
#    $$
# 
# ---
# 
# #### **Model Inference**
# 
# Given a new query $q_{\text{new}}$:
# 
# 1. Compute $\vec{x}_{\text{new}} = [\text{Normalize}(F(q_{\text{new}}));\ \text{Normalize}(E(q_{\text{new}}))]$
# 2. Pass it through the trained MLP:
#    $$
#    \hat{y}_{\text{new}} = f_\theta(\vec{x}_{\text{new}})
#    $$
# 3. Select the model with the highest predicted probability:
#    $$
#    M_{\text{new}} = \arg\max_{j \in \{1, \dots, m\}} \ \hat{y}_{\text{new}}[j]
#    $$
# 
# ---
# 
# #### **Next Steps**
# 
# Explore making $\lambda$ a **trainable parameter** or incorporating user feedback into the soft labels to further improve the self-supervised routing model.
# 
# Create a fallback option if MLP is not sure about the model recommendation.
# 

# # Some Ideas from Dennis echo Justin's Ideas:
# 
# 
# If we choose to use soft labels, one important design question is:
# Should we make the weight parameter λ a learnable gating mechanism that depends on the query and its features?
# Alternatively, should we design a temperature-tuned weighting network that dynamically adjusts soft label strength?
# 
# Another concern is more fundamental: soft labels treat the teacher's prediction distribution (e.g., BERT or LIMBO) as ground truth.
# However, if the teacher model has a systematic bias — such as preferring high-cost or high-latency actions for certain queries or cost ranges — then the student model will inherit this bias, and may even amplify it in rare or tail scenarios.
# 
# That said, the biggest advantage of the soft label solution is its cold-start capability:
# Once a teacher is ready, we can immediately generate soft labels and start training the student model — no need for costly manual labeling.
# 
# My original idea was to merge two-stage distillation into a single end-to-end multi-feature classifier.
# 
# For simplicity, let's assume a single-branch input, where each query 
# q is represented by the following feature vector:
# $$
# \mathbf{x}(q)=
# \bigl[
# \underbrace{\mathbf{e}(q)^{\!\top}}_{\text{BERT}\;d_e},
# \;
# \underbrace{\mathbf{c}(q)^{\!\top}}_{\text{Cluster}\;K},
# \;
# \underbrace{\mathbf{m}(q)^{\!\top}}_{\text{Meta}\;d_m}
# \bigr]^{\!\top}
# \in\mathbb{R}^{d},\qquad
# d=d_e+K+d_m
# $$
# 
# ---
# 
# MLP Structure (shallow for clarity)
# $$
# \begin{aligned}
# \mathbf{h}_1 &= \sigma\!\bigl(W_1\mathbf{x}+\mathbf{b}_1\bigr) \\[2pt]
# \mathbf{h}_2 &= \sigma\!\bigl(W_2\mathbf{h}_1+\mathbf{b}_2\bigr) \\[2pt]
# \boldsymbol{\pi}(q;\theta) &= \operatorname{softmax}\!\bigl(W_3\mathbf{h}_2+\mathbf{b}_3\bigr)
# \;\in[0,1]^{M}
# \end{aligned}
# $$
# 
# Here, $$ \( \pi_j(q;\theta) \) $$ is the predicted probability of selecting model/action \( j \) for query \( q \).
# 
# To encourage disentanglement among different input groups, we can use group-wise L2 or L1 penalties:
# 
# $$
# \mathrm{Reg}(\theta)=
# \lambda_e\lVert W_1^{(e)}\rVert_2^{2}
# +\lambda_c\lVert W_1^{(c)}\rVert_2^{2}
# +\lambda_m\lVert W_1^{(m)}\rVert_2^{2}
# $$
# 
# correspond to the input layers for BERT embeddings, cluster assignment, and meta features, respectively.
# 
# ---
# 
# 
# Loss Function Options
# 
# 1. Cross Entropy (when known winning model/action):
# $$
# \mathcal{L}_{\mathrm{CE}}(\theta)
# = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M}
# y_{ij}\,\log \pi_j(q_i;\theta)
# $$
# plus 
# Expected Utility Maximization:
# 
# $$
# \mathcal{L}_{\mathrm{EU}}(\theta)
# = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M}
# \pi_j(q_i;\theta)\,u_{ij},
# \qquad
# u_{ij}=\beta\,\mathrm{Quality}_{ij}-(1-\beta)\,\mathrm{Cost}_{ij}
# $$
# OR
# 2. Policy‑Gradient ( Bandit, the reward can be obtained by actually running all possible LLMs and getting them evaluated, or this can be used for continues learning)
# $$
# \mathcal{L}_{\mathrm{PG}}(\theta)
# = -\frac{1}{N}\sum_{i=1}^{N}
# \hat{u}_{i a_i}\,
# \log \pi_{a_i}(q_i;\theta)
# $$
# 
# For example:
# 
# 
# Final Objective (if we adopt option 1 and do not consider feedback control for now):
# 
# $$
# \boxed{
# \min_{\theta}\;
# \mathcal{L}(\theta)=
# \alpha\,\mathcal{L}_{\mathrm{CE}}
# +\!(1-\alpha)\,\mathcal{L}_{\mathrm{EU}}
# +\mathrm{Reg}(\theta)
# }
# $$
# 

# In[ ]:





# In[ ]:


# MLP training function
def train_MLP(Q_LIMBO, Q_BERT, M_LIMBO, M_BERT, lambda):
  # MLP training function
  return trained_MLP # trained_MLP is the trained MLP model with its architecutre, weights and biases


# In[ ]:


# MLP inference function
def MLP_inference(query, trained_MLP):
  # MLP inference function
  q_limbo = Normalize(F(query))
  q_bert = Normalize(E(query))
  x = concat(q_limbo, q_bert) # this is the input

  y_pred = trained_MLP(x)
  recommended_model = argmax(y_pred)
  return recommended_model


# # Multi-Head Attention Model

# In[ ]:


def train_attention_router(Q_LIMBO, Q_BERT, M_LIMBO, M_BERT, lambda):
    # Step 1: normalize inputs
    # Step 2: fuse using attention
    # Step 3: train to predict soft label:
    # y_target = lambda * M_LIMBO + (1 - lambda) * M_BERT (same as MLP)
    return trained_model


# In[ ]:


# Inference model should be the same as MLP

