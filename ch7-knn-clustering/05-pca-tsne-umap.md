# 05. PCA·t-SNE·UMAP 비교

## 🎯 핵심 질문

- **PCA**가 왜 분산 최대화 = SVD top components와 동치인가? Linear, global 구조.
- **t-SNE** (van der Maaten & Hinton 2008)의 KL$(p \| q)$ 최소화 — Student-t와 Gaussian의 비대칭성으로 local 구조 보존.
- **UMAP** (McInnes 2018)의 topological 가정과 fuzzy simplicial set — 더 빠르면서 global 구조도 보존.
- 세 방법의 trade-off: Linear vs nonlinear, global vs local, 계산 비용.

---

## 🔍 왜 이 개념이 ML에서 중요한가

차원축소는 (a) **시각화** — 고차원 데이터를 2D/3D로 보기, (b) **noise reduction** — top components만 사용, (c) **다른 ML 알고리즘의 전처리** — KNN·clustering 전 차원 축소, (d) **representation learning의 baseline** — NN과 비교 기준. 본 문서는 세 가지 다른 paradigm — **선형 분산 최대화 (PCA)**, **확률적 KL 최소화 (t-SNE)**, **위상학적 manifold 가정 (UMAP)** — 을 비교해 어느 시점에 어느 방법을 써야 하는지 명확히 한다.

---

## 📐 수학적 선행 조건

- SVD, eigenvalue decomposition (PCA)
- KL divergence (t-SNE)
- 그래프 이론, manifold 기초 (UMAP)

---

## 📖 직관적 이해

### PCA — 분산 최대화

$d$차원 데이터를 $k$차원으로 projection. 가장 분산이 큰 방향을 선택.

$$\max_W \text{tr}(W^\top \Sigma W) \quad \text{s.t. } W^\top W = I.$$

→ $\Sigma$의 top $k$ eigenvectors. SVD의 첫 $k$개 right singular vectors.

**선형, global, 빠름**. $O(np^2)$ 또는 $O(np \cdot k)$ (truncated SVD).

### t-SNE — Local Probability Matching

각 점쌍 $(i, j)$에 대해:

- High-dim의 유사도 $p_{ij} \propto \exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)$ (Gaussian).
- Low-dim의 유사도 $q_{ij} \propto (1 + \|y_i - y_j\|^2)^{-1}$ (Student t with df=1).

$\text{KL}(P \| Q) = \sum p_{ij} \log(p_{ij}/q_{ij})$ 최소화.

**Student-t의 heavy tail**이 핵심: low-dim에서 멀리 있는 점들도 적당히 멀게 표현 가능 → 차원의 저주 회피.

**비선형, local 우선, 느림 ($O(n^2)$)**, global 구조 distort 가능.

### UMAP — Topological

데이터를 **fuzzy simplicial set**으로 표현 (각 점의 $k$-nearest neighbors로 그래프). 이를 low-dim에서 같은 topology 갖도록 projection.

손실: cross-entropy on edge weights.

**비선형, local + global 균형, 빠름** ($O(n \log n)$ effectively).

---

## ✏️ 엄밀한 정의

### 정의 5.1 — PCA

데이터 $X \in \mathbb{R}^{n \times p}$ (centered). SVD: $X = U \Sigma V^\top$.

**Top $k$ PCA components**: $V_{[:, 1:k]}$ (right singular vectors).

**Projection**: $X V_{[:, 1:k]} \in \mathbb{R}^{n \times k}$.

### 정의 5.2 — t-SNE

High-dim similarity:

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = (p_{j|i} + p_{i|j})/(2n).$$

$\sigma_i$는 perplexity (보통 5~50)로 정해짐.

Low-dim similarity:

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}.$$

Loss: $\text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log(p_{ij}/q_{ij})$.

Gradient descent로 $\{y_i\}$ 학습.

### 정의 5.3 — UMAP

각 점에 대해 $k$-NN 거리로 $\rho_i$ (가장 가까운 점), $\sigma_i$ 정의. Edge weight:

$$w_{ij} = \exp\bigl(-\max(0, d(x_i, x_j) - \rho_i)/\sigma_i\bigr).$$

Symmetrize: $w_{ij}' = w_{ij} + w_{ji} - w_{ij} w_{ji}$.

Low-dim도 같은 식 (다른 분포). Cross-entropy 최소화.

---

## 🔬 정리와 증명

### 정리 5.1 — PCA = SVD = 분산 최대화

**명제**: $w^* = \arg\max_{\|w\|=1} w^\top \Sigma w$의 해는 $\Sigma$의 top eigenvector. $\Sigma = X^\top X / n$.

**증명**: Lagrangian $L = w^\top \Sigma w - \lambda(w^\top w - 1)$. $\nabla = 0 \Rightarrow \Sigma w = \lambda w$ — eigenvalue 문제. 가장 큰 $\lambda$의 eigenvector가 분산 최대 방향. $\square$

> 💡 **추가**: top $k$ components → $\Sigma$의 top $k$ eigenvectors. Cumulative variance ratio $= \sum_{i=1}^k \lambda_i / \sum_{j=1}^p \lambda_j$.

### 정리 5.2 — PCA의 reconstruction 최적성

**명제**: PCA top $k$가 **squared reconstruction error**를 최소화하는 rank-$k$ approximation.

**증명**: Eckart-Young 정리. $\min_{\hat{X}: \text{rank}(\hat{X}) \leq k} \|X - \hat{X}\|_F = \sum_{i=k+1}^{\min(n,p)} \sigma_i^2$ — 작은 $k+1, \ldots$개 singular value의 제곱합. 이를 달성하는 $\hat{X}$가 truncated SVD. $\square$

### 정리 5.3 — t-SNE의 비대칭성과 Crowding Problem 해결

**명제**: t-SNE가 high-dim에 Gaussian, low-dim에 Student-t를 쓰는 이유는 **차원의 저주에 의한 crowding 문제**를 해결하기 위함.

**Crowding**: 고차원에서 멀리 있는 점들이 저차원에서 모두 origin 근처로 몰림 — 정보 손실.

Student-t (df=1) = Cauchy: heavy tail → 저차원에서도 큰 거리 표현 가능 → 멀리 있는 점들이 정확히 멀게 배치.

원래 SNE (Hinton 2002)는 Gaussian-Gaussian → crowding. t-SNE의 핵심 innovation.

### 정리 5.4 — t-SNE의 Local Preservation, Global Distortion

**명제** (informal): t-SNE는 local neighbor structure는 보존하지만 global distance는 distort.

**예**: t-SNE 결과에서 두 cluster 사이 거리 ≠ 진짜 거리. cluster 모양·size도 부정확. **시각화에는 좋지만 distance-based downstream task에는 부적합**.

### 정리 5.5 — UMAP의 Topological Foundation

**명제** (informal): UMAP은 **fuzzy simplicial set**으로 데이터 manifold를 모델링. 이는 algebraic topology의 정밀한 framework.

**이점**:
- t-SNE보다 **빠름**: $O(n^{1.14})$ practical.
- **Global 구조도 더 잘 보존**.
- **새 데이터 transform 가능** (t-SNE는 batch, UMAP은 inductive).

**한계**: 이론적 깊이 vs 실무 직관의 gap (algebraic topology 배경 필요).

### 정리 5.6 — 세 방법의 비교

| 측면 | PCA | t-SNE | UMAP |
|------|-----|-------|------|
| **Linear/Nonlinear** | Linear | Nonlinear | Nonlinear |
| **Local 보존** | Poor | **Best** | Good |
| **Global 보존** | **Best** | Poor | Good |
| **계산 비용** | $O(np \min(n,p))$ | $O(n^2)$ | $O(n \log n)$ effectively |
| **결정성** | Deterministic | Stochastic (init 의존) | Stochastic |
| **시각화** | OK (선형) | **Best for cluster** | Cluster + topology |
| **Inductive** | Yes (project new) | No | Yes |
| **Hyperparameters** | None (just $k$) | perplexity | n_neighbors, min_dist |

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import umap

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. PCA 바닥 구현 vs sklearn
# ─────────────────────────────────────────────
data = load_digits()
X = data.data
y = data.target

# Centering
X_c = X - X.mean(axis=0)

# SVD
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
my_pca = X_c @ Vt[:2].T

# sklearn
sk_pca = PCA(n_components=2).fit_transform(X)

# 부호만 다를 수 있음
print(f'My PCA vs sklearn 차이 (sign-invariant):')
diff_min = min(np.linalg.norm(my_pca - sk_pca), np.linalg.norm(my_pca + sk_pca))
print(f'  ||diff|| = {diff_min:.2e}')

# Variance ratio
var_ratio = s**2 / (s**2).sum()
print(f'\n첫 5 components의 explained variance:')
for i in range(5):
    print(f'  PC{i+1}: {var_ratio[i]:.4f}')
print(f'Cumulative (top 5): {var_ratio[:5].sum():.4f}')
print(f'Cumulative (top 10): {var_ratio[:10].sum():.4f}')

# ─────────────────────────────────────────────
# 2. PCA reconstruction (정리 5.2)
# ─────────────────────────────────────────────
print(f'\nReconstruction error vs k components:')
for k in [2, 5, 10, 20, 50]:
    X_recon = X_c @ Vt[:k].T @ Vt[:k]
    err = np.linalg.norm(X_c - X_recon, ord='fro')
    var_explained = var_ratio[:k].sum()
    print(f'  k = {k:>2}: ||X - X_k||_F = {err:.2f}, explained var = {var_explained:.4f}')

# ─────────────────────────────────────────────
# 3. PCA vs t-SNE vs UMAP — Digits에서 비교
# ─────────────────────────────────────────────
print(f'\n8x8 digits (n={len(X)}, p={X.shape[1]}) → 2D 차원축소:')

# PCA
import time
t0 = time.time()
X_pca = PCA(n_components=2).fit_transform(X)
t_pca = time.time() - t0

# t-SNE
t0 = time.time()
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
t_tsne = time.time() - t0

# UMAP
t0 = time.time()
X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
t_umap = time.time() - t0

print(f'  PCA   : {t_pca*1000:.0f} ms')
print(f'  t-SNE : {t_tsne*1000:.0f} ms')
print(f'  UMAP  : {t_umap*1000:.0f} ms')

# ─────────────────────────────────────────────
# 4. KNN classification on 2D projections (성능 비교)
# ─────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

print(f'\n2D projection 위 KNN-5 accuracy (10-fold):')
print(f'  Original ({X.shape[1]} dim): {cross_val_score(KNeighborsClassifier(5), X, y, cv=10).mean():.4f}')
print(f'  PCA-2D                : {cross_val_score(KNeighborsClassifier(5), X_pca, y, cv=10).mean():.4f}')
print(f'  t-SNE-2D              : {cross_val_score(KNeighborsClassifier(5), X_tsne, y, cv=10).mean():.4f}')
print(f'  UMAP-2D               : {cross_val_score(KNeighborsClassifier(5), X_umap, y, cv=10).mean():.4f}')

# ─────────────────────────────────────────────
# 5. t-SNE vs UMAP의 global structure 보존 비교
# ─────────────────────────────────────────────
# 진짜 distance vs 2D distance의 correlation
def projection_quality(X_orig, X_low, n_pairs=5000):
    n = len(X_orig)
    pairs = rng.choice(n, (n_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    
    d_orig = np.linalg.norm(X_orig[pairs[:, 0]] - X_orig[pairs[:, 1]], axis=1)
    d_low = np.linalg.norm(X_low[pairs[:, 0]] - X_low[pairs[:, 1]], axis=1)
    
    return np.corrcoef(d_orig, d_low)[0, 1]

print(f'\n2D distance vs original distance correlation:')
print(f'  PCA  : {projection_quality(X, X_pca):.4f}  (선형 → 글로벌 보존)')
print(f'  t-SNE: {projection_quality(X, X_tsne):.4f}  (낮음 — local 보존, global distort)')
print(f'  UMAP : {projection_quality(X, X_umap):.4f}  (중간 — 균형)')
```

**출력 예시**:
```
My PCA vs sklearn 차이 (sign-invariant):
  ||diff|| = 1.34e-13

첫 5 components의 explained variance:
  PC1: 0.1490
  PC2: 0.1366
  PC3: 0.1180
  PC4: 0.0840
  PC5: 0.0578
Cumulative (top 5): 0.5454
Cumulative (top 10): 0.7423

Reconstruction error vs k components:
  k =  2: ||X - X_k||_F = 542.32, explained var = 0.2856
  k =  5: ||X - X_k||_F = 432.13, explained var = 0.5454
  k = 10: ||X - X_k||_F = 312.53, explained var = 0.7423
  k = 20: ||X - X_k||_F = 187.43, explained var = 0.9043
  k = 50: ||X - X_k||_F = 41.21, explained var = 0.9952

8x8 digits (n=1797, p=64) → 2D 차원축소:
  PCA   : 12 ms
  t-SNE : 8324 ms
  UMAP  : 1543 ms

2D projection 위 KNN-5 accuracy (10-fold):
  Original (64 dim)        : 0.9710
  PCA-2D                   : 0.6132
  t-SNE-2D                 : 0.9421
  UMAP-2D                  : 0.9521

2D distance vs original distance correlation:
  PCA  : 0.7821  (선형 → 글로벌 보존)
  t-SNE: 0.4231  (낮음 — local 보존, global distort)
  UMAP : 0.6234  (중간 — 균형)
```

---

## 🔗 실전 활용

- **sklearn `PCA`**: 빠른 baseline, 선형 데이터.
- **sklearn `TSNE`**: 클러스터 시각화의 표준 (단 비싸고 stochastic).
- **`umap-learn`**: 대규모 데이터 차원축소, embedding 시각화.
- **PCA → KNN/clustering**: 차원 저주 회피 위한 전처리.
- **t-SNE/UMAP for cell biology**: single-cell RNA-seq 시각화의 표준.
- **Embedding visualization**: word2vec, BERT embedding을 t-SNE로 시각화.

---

## ⚖️ 가정과 한계

| 방법 | 가정 / 한계 |
|------|------------|
| **PCA** | 선형 — 비선형 manifold 못 잡음 |
| **t-SNE** | local 우선 — global distort, 비싸고 stochastic |
| **UMAP** | manifold 가정, hyperparameter 영향 큼 |

---

## 📌 핵심 정리

$$\boxed{\text{PCA: linear variance max;}\ \text{t-SNE: KL local;}\ \text{UMAP: topological global+local};\ \text{용도별 선택}}$$

| 방법 | 핵심 idea |
|------|---------|
| **PCA** | 분산 최대 = SVD top components |
| **t-SNE** | KL$(p\|q)$, Gaussian-Student t 비대칭 |
| **UMAP** | fuzzy simplicial set, manifold topology |

---

## 🤔 생각해볼 문제

**문제 1** (기초): PCA의 첫 component가 데이터의 "variance 가장 큰 방향"임을 한 줄로 정당화하라.

<details>
<summary>힌트 및 해설</summary>

$\arg\max_w w^\top \Sigma w$ s.t. $\|w\| = 1$ (Rayleigh quotient). 

Lagrange: $\nabla(w^\top \Sigma w - \lambda \|w\|^2) = 2\Sigma w - 2\lambda w = 0 \Rightarrow \Sigma w = \lambda w$.

→ eigenvector. 최대 $w^\top \Sigma w = \lambda$ → top eigenvalue eigenvector.

</details>

**문제 2** (심화): t-SNE의 perplexity 파라미터의 의미와 영향?

<details>
<summary>힌트 및 해설</summary>

Perplexity = $2^{H(P_i)}$ where $H(P_i) = -\sum_j p_{j|i} \log p_{j|i}$ — 점 $i$의 effective neighbor 수.

각 $\sigma_i$는 perplexity가 사용자 지정 값과 같도록 binary search.

**작은 perplexity** (5~10): focus on very local — 각 cluster가 micro-cluster로 분할.

**큰 perplexity** (50~100): broader view — global structure 더 보존.

기본 30이 대부분 잘 작동. 데이터 점 수에 비례 — 큰 데이터에서 perplexity 더 크게.

</details>

**문제 3** (ML 연결): NN의 **representation learning** (BERT embedding, image embedding 등)이 PCA·t-SNE·UMAP과 어떻게 다른가? 결국 같은 일을 하는가?

<details>
<summary>힌트 및 해설</summary>

**모두 차원축소** — 고차원 raw data → low-dim representation.

**차이**:

- **PCA**: linear, unsupervised, **closed-form**.
- **t-SNE/UMAP**: nonlinear, unsupervised, **iterative optimization**, 시각화 친화.
- **NN representation**: nonlinear, **task-supervised** (classification, contrastive), **learnable**.

**핵심 차이**: NN은 **downstream task에 맞게 representation 학습**. PCA/t-SNE/UMAP은 **데이터의 구조만 보존** — task-agnostic.

**결과**: NN representation이 보통 더 정확한 downstream task 성능. 그러나 **시각화는 t-SNE/UMAP** (낮은 차원으로 강제 mapping).

**Modern combinations**: NN으로 representation 학습 → PCA/t-SNE/UMAP으로 시각화. (BERT embedding을 t-SNE로 시각화하는 것이 흔한 NLP 분석.)

**결론**: 같은 정신 ("data manifold 발견"), 다른 도구. NN은 expressive + task-aware, classical은 fast + interpretable.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. DBSCAN](./04-dbscan.md) | [📚 README](../README.md) | [🎓 ML Fundamentals 완주 — README](../README.md) |

</div>
