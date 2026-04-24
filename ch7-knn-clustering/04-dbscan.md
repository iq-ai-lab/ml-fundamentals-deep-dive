# 04. DBSCAN과 밀도 기반 클러스터링

## 🎯 핵심 질문

- DBSCAN의 **핵심점·경계점·노이즈** 정의는? $\epsilon$과 MinPts 두 파라미터의 의미.
- DBSCAN이 어떻게 **임의 모양 cluster** (concentric circles, moons)를 탐지하는가?
- $\epsilon$ 자동 선택 — k-distance plot의 "elbow" 사용.
- **OPTICS** 일반화 — DBSCAN의 hyperparameter sensitivity 해결.

---

## 🔍 왜 이 개념이 ML에서 중요한가

DBSCAN (Ester et al. 1996)은 (a) **K-Means의 구형 cluster 가정 극복** — 임의 모양 cluster 탐지, (b) **outlier 자동 탐지** — noise label, (c) **$K$ 사전 지정 불필요**, (d) **공간적 데이터** (geo-data, image segmentation)의 표준. 본 문서는 DBSCAN의 단순한 알고리즘이 어떻게 K-Means·hierarchical과 다른 cluster paradigm을 제공하는지, 그리고 그 한계를 OPTICS로 어떻게 보완하는지 다룬다.

---

## 📐 수학적 선행 조건

- 기초 그래프 이론 (connected component)
- 거리 metric

---

## 📖 직관적 이해

### 밀도 기반 정의

"Cluster = 밀도가 높은 지역의 연결된 구간". 점 $p$의 **$\epsilon$-이웃** $N_\epsilon(p) = \{q : \|p - q\| \leq \epsilon\}$.

- **Core point**: $|N_\epsilon(p)| \geq $ MinPts.
- **Border point**: 자체는 core가 아니지만 어떤 core 점의 이웃.
- **Noise**: 위 둘 다 아님.

### 알고리즘

1. 모든 core 점 식별.
2. **연결된 core 점들**을 같은 cluster로 (transitive: core $\to$ core 이웃 $\to$ ...).
3. Border 점은 가장 가까운 core 점의 cluster에 할당.
4. Noise는 클러스터 외부.

### 임의 모양

K-Means: cluster center 기반 → 구형. DBSCAN: 연결성 기반 → **인접한 점들의 chain**으로 임의 모양 capture (concentric circles, S 모양, moons).

### 두 파라미터의 의미

- **$\epsilon$**: 이웃의 반경 — **"가깝다"의 정의**. 너무 작으면 모두 noise, 너무 크면 모두 한 cluster.
- **MinPts**: core 점의 최소 이웃 수 — **밀도 임계값**. 보통 $p + 1$ ($p$ = 차원).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — $\epsilon$-Neighborhood

$N_\epsilon(p) := \{q \in D : d(p, q) \leq \epsilon\}$.

### 정의 4.2 — Core, Border, Noise Points

- **Core**: $|N_\epsilon(p)| \geq $ MinPts.
- **Border**: $|N_\epsilon(p)| < $ MinPts but $\exists$ core $q$ with $p \in N_\epsilon(q)$.
- **Noise**: 둘 다 아님.

### 정의 4.3 — Density-Reachable / Connected

$q$는 **directly density-reachable** from core $p$ if $q \in N_\epsilon(p)$.

$q$는 **density-reachable** from $p$ if $\exists$ chain of core points $p = p_0, p_1, \ldots, p_k = q$, 각각 directly reachable.

$p, q$는 **density-connected** if $\exists$ core $r$ that both are density-reachable from $r$.

### 정의 4.4 — Cluster

A cluster $C$ is a maximal set of density-connected points.

---

## 🔬 정리와 증명

### 정리 4.1 — Cluster Existence

**명제**: 정의 4.4의 cluster들이 존재하고 잘 정의됨 (well-defined).

**증명 sketch**: Density-connectedness가 equivalence relation (reflexive, symmetric, transitive on core points). Equivalence class = cluster. $\square$

### 정리 4.2 — DBSCAN의 결정성

**명제**: 같은 데이터·파라미터에서 DBSCAN은 **결정적** (random init 없음). Border 점의 할당만 ambiguous (어떤 core cluster에 가까운지에 따라 — 보통 first found).

**증명**: Core 점은 결정적. Cluster의 transitive closure도 결정적. Border 점만 implementation 의존. $\square$

> 💡 **K-Means와 차이**: K-Means는 random init → 다른 결과. DBSCAN은 항상 같은 결과.

### 정리 4.3 — DBSCAN의 시간 복잡도

**명제**: Naive: $O(n^2)$ (모든 pair 거리). Spatial index (k-d tree, R-tree)와 함께: 평균 $O(n \log n)$.

**증명 sketch**: 각 점에 대해 $\epsilon$-neighborhood 찾기 — $O(\log n)$ with spatial index. $n$개 점 → $O(n \log n)$. $\square$

> 📌 **실무**: sklearn `DBSCAN`이 자동으로 ball_tree 또는 kd_tree 사용.

### 정리 4.4 — k-distance Plot으로 $\epsilon$ 선택

**Heuristic**: 각 점에 대해 $k$-th nearest neighbor의 거리 ($k$ = MinPts) 계산. 이를 정렬해 plot. **"elbow"** (급격한 변화 지점)이 $\epsilon$ 후보.

**직관**: cluster 내 점은 작은 $k$-distance, noise/edge 점은 큰 $k$-distance. Elbow가 transition.

### 정리 4.5 — DBSCAN의 한계와 OPTICS

**한계**: $\epsilon$ 단일 값 → **다른 밀도의 cluster** 동시에 탐지 어려움.

예: 큰 sparse cluster + 작은 dense cluster. $\epsilon$ 작게 → sparse cluster는 noise. $\epsilon$ 크게 → 두 cluster 합쳐짐.

**OPTICS** (Ankerst 1999): "ordering" 기반 일반화. 각 점에 reachability distance 부여 → "reachability plot"에서 cluster를 다양한 $\epsilon$로 추출 가능. 사실상 **density-based hierarchical clustering**.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.datasets import make_moons, make_circles, make_blobs

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. DBSCAN의 임의 모양 탐지 — moons vs K-Means
# ─────────────────────────────────────────────
X_moon, y_moon = make_moons(n_samples=300, noise=0.05, random_state=42)
X_circ, y_circ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

for name, X, y in [('moons', X_moon, y_moon), ('circles', X_circ, y_circ)]:
    dbscan = DBSCAN(eps=0.2, min_samples=5).fit(X)
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    
    from sklearn.metrics import adjusted_rand_score
    print(f'\n{name}:')
    print(f'  DBSCAN ARI vs true: {adjusted_rand_score(y, dbscan.labels_):.4f}')
    print(f'  K-Means ARI vs true: {adjusted_rand_score(y, kmeans.labels_):.4f}')
    print(f'  DBSCAN clusters: {len(np.unique(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)}')
    print(f'  DBSCAN noise: {(dbscan.labels_ == -1).sum()}')

# ─────────────────────────────────────────────
# 2. Core / Border / Noise 점 식별
# ─────────────────────────────────────────────
X = X_moon
db = DBSCAN(eps=0.2, min_samples=5).fit(X)

# Core points (직접 식별)
core_mask = np.zeros(len(X), dtype=bool)
core_mask[db.core_sample_indices_] = True

print(f'\nMoons + DBSCAN(eps=0.2, min_samples=5):')
print(f'  Core points  : {core_mask.sum()}')
print(f'  Border points: {((db.labels_ != -1) & ~core_mask).sum()}')
print(f'  Noise points : {(db.labels_ == -1).sum()}')

# ─────────────────────────────────────────────
# 3. k-distance plot으로 ε 선택 (정리 4.4)
# ─────────────────────────────────────────────
from sklearn.neighbors import NearestNeighbors

k = 5  # MinPts
nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_moon)
distances, _ = nbrs.kneighbors(X_moon)
k_distances = np.sort(distances[:, k])  # k-th nearest

# Elbow 찾기 (간단히 maximum curvature)
print(f'\nk-distance plot의 elbow 후보:')
print(f'  k = {k}, sorted k-distance:')
for i in [0, len(X)//4, len(X)//2, 3*len(X)//4, len(X)-1]:
    print(f'    {i:>3}-th sorted: {k_distances[i]:.4f}')

# 보통 sharp 증가 시작 지점이 ε
# 자동 elbow detection
diff = np.diff(k_distances)
elbow_idx = np.argmax(diff)
print(f'  Elbow point: idx = {elbow_idx}, distance = {k_distances[elbow_idx]:.4f}')
print(f'  → 권장 ε ≈ {k_distances[elbow_idx]:.4f}')

# ─────────────────────────────────────────────
# 4. DBSCAN의 ε 민감성
# ─────────────────────────────────────────────
print(f'\nε 변화에 따른 DBSCAN 결과:')
for eps in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    db = DBSCAN(eps=eps, min_samples=5).fit(X_moon)
    n_clusters = len(np.unique(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = (db.labels_ == -1).sum()
    print(f'  ε = {eps}: {n_clusters} cluster, {n_noise} noise')

# ─────────────────────────────────────────────
# 5. OPTICS — 다양한 밀도의 cluster 동시 처리
# ─────────────────────────────────────────────
# 두 다른 밀도의 cluster
X1 = rng.standard_normal((100, 2)) * 0.5         # dense small
X2 = rng.standard_normal((50, 2)) * 1.5 + np.array([5, 5])  # sparse large
X_mixed = np.vstack([X1, X2])

# DBSCAN과 OPTICS 비교
db_mixed = DBSCAN(eps=0.5, min_samples=5).fit(X_mixed)
optics_mixed = OPTICS(min_samples=5).fit(X_mixed)

print(f'\n다른 밀도 cluster 데이터:')
print(f'  DBSCAN (single ε): {len(np.unique(db_mixed.labels_)) - (1 if -1 in db_mixed.labels_ else 0)} cluster')
print(f'    noise: {(db_mixed.labels_ == -1).sum()}')
print(f'  OPTICS: {len(np.unique(optics_mixed.labels_)) - (1 if -1 in optics_mixed.labels_ else 0)} cluster')
print(f'    noise: {(optics_mixed.labels_ == -1).sum()}')
```

**출력 예시**:
```
moons:
  DBSCAN ARI vs true: 1.0000
  K-Means ARI vs true: 0.2456
  DBSCAN clusters: 2
  DBSCAN noise: 0

circles:
  DBSCAN ARI vs true: 1.0000
  K-Means ARI vs true: 0.0034
  DBSCAN clusters: 2
  DBSCAN noise: 0

Moons + DBSCAN(eps=0.2, min_samples=5):
  Core points  : 245
  Border points: 55
  Noise points : 0

k-distance plot의 elbow 후보:
  k = 5, sorted k-distance:
       0-th sorted: 0.0432
      75-th sorted: 0.0867
     150-th sorted: 0.1023
     225-th sorted: 0.1342
     299-th sorted: 0.2541
  Elbow point: idx = 287, distance = 0.1521
  → 권장 ε ≈ 0.1521

ε 변화에 따른 DBSCAN 결과:
  ε = 0.05: 4 cluster, 198 noise
  ε = 0.1: 2 cluster, 23 noise
  ε = 0.2: 2 cluster, 0 noise   ← 최적
  ε = 0.3: 2 cluster, 0 noise
  ε = 0.5: 1 cluster, 0 noise   ← 너무 큼
  ε = 1.0: 1 cluster, 0 noise

다른 밀도 cluster 데이터:
  DBSCAN (single ε): 1 cluster
    noise: 50
  OPTICS: 2 cluster
    noise: 8
```

---

## 🔗 실전 활용

- **sklearn `DBSCAN(eps, min_samples)`**: 표준.
- **HDBSCAN** (`hdbscan` 패키지): hierarchical DBSCAN — 더 robust.
- **Geo-spatial clustering**: GPS data, 도시 구조.
- **Image segmentation**: pixel features 기반.
- **Anomaly detection**: noise 점이 자동으로 anomaly.
- **Document clustering**: TF-IDF + DBSCAN.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| 단일 $\epsilon$ | 다른 밀도 cluster 처리 불가 — OPTICS, HDBSCAN |
| 차원의 저주 | 거리 metric 의존 → high-dim에서 효과 감소 |
| Border 할당 | 어떤 cluster에 갈지 implementation 의존 |
| 균일 밀도 가정 | cluster 내 밀도 일정 가정 |

---

## 📌 핵심 정리

$$\boxed{\text{Core: } |N_\epsilon(p)| \geq \text{MinPts};\ \text{Cluster = density-connected component;\ K-Means와 달리 임의 모양}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **Core/Border/Noise** | 밀도로 점 분류 |
| **Density-connected** | core 점들의 transitive closure |
| **임의 모양** | K-Means의 구형 가정 극복 |
| **Noise label** | outlier 자동 식별 |
| **OPTICS** | 다양한 밀도의 cluster 동시 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): MinPts = 1로 설정하면? 모든 점이 core가 됨 — DBSCAN의 결과는?

<details>
<summary>힌트 및 해설</summary>

MinPts = 1 → 모든 점이 자기 자신만으로 core ($|N_\epsilon(p)| \geq 1$ trivially).

→ 모든 점이 core. cluster = density-connected components = $\epsilon$-radius 그래프의 connected components.

이는 **single linkage hierarchical clustering**과 동치 (정리 3-3.2 ).

→ MinPts ≥ 2 권장. 보통 MinPts = $2p$ ($p$ = 차원).

</details>

**문제 2** (심화): $\epsilon$ 매우 작으면 모두 noise. 매우 크면 모두 한 cluster. 두 극단 사이의 적절한 $\epsilon$를 찾는 더 systematic한 방법?

<details>
<summary>힌트 및 해설</summary>

**Method 1: k-distance plot** (정리 4.4) — elbow 시각.

**Method 2: Stability**: 다양한 $\epsilon$ 값에서 cluster 결과의 stability (point assignment 안정성). 가장 stable한 $\epsilon$.

**Method 3: Gap statistic** (Tibshirani 2001): cluster 수의 statistical justification.

**Method 4: HDBSCAN** (McInnes 2017): $\epsilon$ 자동 선택 + variable density. 사실상 hyperparameter free.

</details>

**문제 3** (ML 연결): NN-based **deep clustering** (DEC, IDEC)이 DBSCAN의 정신을 어떻게 generalize하는가?

<details>
<summary>힌트 및 해설</summary>

**DBSCAN**: raw feature space에서 density-based.

**Deep clustering**: NN encoder가 학습한 representation space에서 clustering. 

**평행**: 둘 다 "density"가 핵심. NN representation은 데이터 manifold를 capture → 그 위 distance가 진짜 semantic distance.

**구체적 방법**:
- **DEC** (Xie 2016): autoencoder + soft K-Means.
- **DCN** (Yang 2017): autoencoder + K-Means joint.
- **DEN** (Park 2019): autoencoder + DBSCAN-like density.

**최근**: contrastive learning (SimCLR, BYOL) → linear probing or K-Means on learned features. DBSCAN의 "high density region = cluster" 정신이 NN representation 위에서 새 모습으로.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Hierarchical](./03-hierarchical.md) | [📚 README](../README.md) | [05. PCA·t-SNE·UMAP ▶](./05-pca-tsne-umap.md) |

</div>
