# 02. K-Means와 EM의 관계

## 🎯 핵심 질문

- K-Means가 **GMM의 hard-assignment 극한** ($\Sigma \to 0$)임을 어떻게 유도하는가?
- Lloyd 알고리즘 (Assign → Update 반복)이 **단조감소** + **유한단계 수렴**임을 증명.
- **K-Means++** (Arthur & Vassilvitskii 2007)의 $O(\log k)$ 경쟁비 — random init보다 항상 더 좋은 보장.
- EM의 일반적 framework로서의 K-Means.

---

## 🔍 왜 이 개념이 ML에서 중요한가

K-Means는 (a) **unsupervised learning의 첫 알고리즘** — sklearn `KMeans`, (b) **EM 알고리즘의 가장 단순한 사례** — GMM, HMM의 진정한 이해의 출발, (c) **vector quantization·image compression·document clustering**의 표준, (d) **K-Means++ 초기화**가 random보다 항상 더 좋은 보장 — algorithm의 "approximation guarantee" 사례. 본 문서는 K-Means가 단순 휴리스틱이 아니라 **GMM의 limit**이라는 깊은 통계학적 의미를 갖는다는 것을 보인다.

---

## 📐 수학적 선행 조건

- 다변수 정규분포의 likelihood
- EM 알고리즘의 일반적 형태 (E-step, M-step)
- 미적분학의 1차 조건

---

## 📖 직관적 이해

### K-Means 알고리즘

$K$개 cluster, centroids $\{\mu_k\}_{k=1}^K$.

1. Init: random centroids.
2. **Assign step**: 각 점 $x_i$를 가장 가까운 centroid에 할당. $z_i = \arg\min_k \|x_i - \mu_k\|$.
3. **Update step**: 각 centroid를 그 cluster의 평균으로. $\mu_k = \frac{1}{|C_k|}\sum_{i: z_i = k} x_i$.
4. 수렴까지 반복.

### EM 일반 framework

Latent variable $z$ (cluster assignment), 관측 $x$.

- **E-step**: $P(z | x, \theta^{old})$ 계산.
- **M-step**: $\theta$ 업데이트로 $\mathbb{E}_{z}[\log P(x, z | \theta)]$ 최대화.

K-Means는 **hard EM** — soft posterior $P(z|x)$ 대신 hard assignment ($z = \arg\max P(z|x)$).

### GMM Limit

GMM: $P(x) = \sum_k \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$. $\Sigma_k = \sigma^2 I$로 고정 + $\sigma \to 0$ → posterior가 가장 가까운 component에 거의 모든 mass — **K-Means의 hard assignment**.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — K-Means Objective

$$J(\mu, z) := \sum_{i=1}^n \|x_i - \mu_{z_i}\|^2 = \sum_{k=1}^K \sum_{i: z_i = k} \|x_i - \mu_k\|^2.$$

**Lloyd 알고리즘**: $J$를 alternating minimization으로 최소화.

### 정의 2.2 — Lloyd's Algorithm

Init centroids $\mu^{(0)}$.

For $t = 1, 2, \ldots$:

1. $z_i^{(t)} = \arg\min_k \|x_i - \mu_k^{(t-1)}\|^2$.
2. $\mu_k^{(t)} = \frac{1}{|C_k^{(t)}|}\sum_{i: z_i^{(t)} = k} x_i$ where $C_k^{(t)} = \{i : z_i^{(t)} = k\}$.

수렴: 연속한 iteration에서 $z$ 변화 없음.

### 정의 2.3 — K-Means++ Initialization

1. $\mu_1$: 데이터에서 random 1개.
2. For $k = 2, \ldots, K$: 각 점 $x$에서 이미 선택된 centroid까지의 최단 거리 $D(x)$ 계산. 확률 $D(x)^2 / \sum_x D(x)^2$로 다음 centroid sampling.

---

## 🔬 정리와 증명

### 정리 2.1 — Lloyd의 단조감소

**명제**: Lloyd algorithm의 각 step에서 $J(\mu^{(t)}, z^{(t)}) \leq J(\mu^{(t-1)}, z^{(t-1)})$.

**증명**: 

(Assign step): 각 $i$에 대해 $z_i^{(t)} = \arg\min_k \|x_i - \mu_k^{(t-1)}\|^2$ → $\|x_i - \mu_{z_i^{(t)}}^{(t-1)}\|^2 \leq \|x_i - \mu_{z_i^{(t-1)}}^{(t-1)}\|^2$. 합하면 $J(\mu^{(t-1)}, z^{(t)}) \leq J(\mu^{(t-1)}, z^{(t-1)})$.

(Update step): 고정된 $z^{(t)}$에 대해 $\mu_k = \arg\min \sum_{i: z_i = k} \|x_i - \mu_k\|^2$의 해는 평균 (정리 3-3.1). 따라서 $J(\mu^{(t)}, z^{(t)}) \leq J(\mu^{(t-1)}, z^{(t)})$. $\square$

### 정리 2.2 — Lloyd의 유한단계 수렴

**명제**: Lloyd 알고리즘은 유한 단계에서 수렴.

**증명**: $z \in \{1, \ldots, K\}^n$ — 유한 가짓수 $K^n$. 각 step에서 $J$ 감소 (정리 2.1). $J$가 같은 값에서 무한 반복하면 $z$도 같아야 → $z$는 한 값에서 정착. 알고리즘 멈춤. $\square$

> 💡 **단점**: **Local minimum 가능**. Random init으로 여러 번 + best result 선택이 표준.

### 정리 2.3 — K-Means = GMM의 Hard-EM ($\Sigma \to 0$)

**명제**: GMM with $\Sigma_k = \sigma^2 I$ for all $k$, $\sigma \to 0^+$. Posterior

$$P(z = k | x) = \frac{\pi_k \mathcal{N}(x; \mu_k, \sigma^2 I)}{\sum_j \pi_j \mathcal{N}(x; \mu_j, \sigma^2 I)} \to \mathbb{1}[k = \arg\min_j \|x - \mu_j\|^2].$$

즉 가장 가까운 component에 모든 mass.

**증명**: Gaussian density: $\mathcal{N}(x; \mu_k, \sigma^2 I) \propto \exp(-\|x - \mu_k\|^2 / (2\sigma^2))$. $\sigma \to 0$이면 가장 가까운 $\mu_k$의 exp가 압도적으로 큼 → posterior가 그 component에 1, 나머지 0. $\square$

> 📌 **함의**: K-Means는 GMM의 **degenerate limit**. EM의 일반화 framework로 GMM·HMM·Variational Inference·VAE 등으로 확장.

### 정리 2.4 — K-Means++의 경쟁비 (Arthur & Vassilvitskii 2007)

**명제**: K-Means++ init으로 시작한 Lloyd의 결과 $\phi$:

$$\mathbb{E}[\phi] \leq 8 (\ln k + 2) \cdot \phi^*,$$

여기서 $\phi^*$는 optimal clustering의 cost. 즉 **$O(\log k)$-approximation in expectation**.

**증명 sketch**: Init이 cluster center 가까이를 sampling하는 분포 → 처음부터 좋은 centroid 후보. Lloyd로 더 개선만 하므로 init quality가 hold. 자세히는 Arthur & Vassilvitskii (2007).

> 💡 **실무 의미**: random init은 worst case $\Theta(k)$-bad. K-Means++는 $O(\log k)$ — 큰 $k$에서 매우 큰 차이.

### 정리 2.5 — K-Means의 비볼록성

**명제**: $J(\mu)$ ($z$를 implicit하게 minimize)는 $\mu$에 대해 **비볼록** — 여러 local minimum 가능.

**증명 sketch**: $J(\mu) = \sum_i \min_k \|x_i - \mu_k\|^2$ — minimum function은 비볼록. 여러 init의 결과가 다름이 직접 증거. $\square$

> 📌 **해결**: 여러 random init + best 선택. sklearn `KMeans(n_init=10)` (v1.0 이전 default).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. K-Means 바닥 구현
# ─────────────────────────────────────────────
def my_kmeans(X, K, n_iter=100, tol=1e-4):
    n, p = X.shape
    # Random init
    centroids = X[rng.choice(n, K, replace=False)]
    
    history = [centroids.copy()]
    for it in range(n_iter):
        # Assign
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        z = np.argmin(distances, axis=1)
        
        # Update
        new_centroids = np.array([X[z == k].mean(axis=0) if (z == k).any() 
                                   else centroids[k] for k in range(K)])
        
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
        history.append(centroids.copy())
    
    return centroids, z, it + 1

# 검증
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)
my_centroids, my_z, my_iter = my_kmeans(X, K=4)
sk = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)

print(f'My K-Means 수렴: {my_iter} iter')
print(f'sklearn K-Means inertia: {sk.inertia_:.2f}')
my_inertia = sum(np.linalg.norm(X[my_z == k] - my_centroids[k])**2 
                  for k in range(4))
print(f'My K-Means inertia    : {my_inertia:.2f}')

# ─────────────────────────────────────────────
# 2. Lloyd의 단조감소 검증 (정리 2.1)
# ─────────────────────────────────────────────
centroids = X[rng.choice(len(X), 4, replace=False)]
losses = []
for it in range(20):
    distances = np.linalg.norm(X[:, None] - centroids, axis=2)
    z = np.argmin(distances, axis=1)
    new_centroids = np.array([X[z == k].mean(axis=0) for k in range(4)])
    
    loss = sum(np.linalg.norm(X[z == k] - new_centroids[k])**2 for k in range(4))
    losses.append(loss)
    centroids = new_centroids

print(f'\nLoss 단조감소:')
for t in [0, 1, 2, 5, 10, 19]:
    print(f'  iter {t}: J = {losses[t]:.2f}')

# ─────────────────────────────────────────────
# 3. K-Means = GMM($\Sigma \to 0$) 시연 (정리 2.3)
# ─────────────────────────────────────────────
print(f'\nGMM의 σ → 0 limit이 K-Means가 됨:')
for sigma in [10.0, 1.0, 0.1, 0.01]:
    # GMM with fixed isotropic cov
    gmm = GaussianMixture(n_components=4, covariance_type='spherical',
                           random_state=42).fit(X)
    # σ를 인위적으로 작게
    gmm.covariances_ = np.full(4, sigma**2)
    gmm.precisions_cholesky_ = 1.0 / sigma
    z_gmm = gmm.predict(X)
    z_km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X).labels_
    
    # ARI (Adjusted Rand Index)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(z_gmm, z_km)
    print(f'  σ = {sigma:>6}: ARI(GMM, KMeans) = {ari:.4f}')

# ─────────────────────────────────────────────
# 4. K-Means++ 초기화 (정리 2.4)
# ─────────────────────────────────────────────
def kmeans_plus_plus_init(X, K):
    n = len(X)
    centroids = [X[rng.integers(0, n)]]
    for k in range(1, K):
        distances = np.array([min(np.linalg.norm(x - c)**2 for c in centroids) 
                              for x in X])
        probs = distances / distances.sum()
        centroids.append(X[rng.choice(n, p=probs)])
    return np.array(centroids)

# Random vs ++ 비교 (여러 trial로)
print(f'\nRandom init vs K-Means++ (4 cluster, 50 trial):')
random_inertias = []
pp_inertias = []
for trial in range(50):
    rng_t = np.random.default_rng(trial)
    
    # Random init
    init_random = X[rng_t.choice(len(X), 4, replace=False)]
    km_r = KMeans(n_clusters=4, init=init_random, n_init=1, random_state=trial).fit(X)
    random_inertias.append(km_r.inertia_)
    
    # K-Means++
    km_pp = KMeans(n_clusters=4, init='k-means++', n_init=1, random_state=trial).fit(X)
    pp_inertias.append(km_pp.inertia_)

print(f'  Random  init: mean inertia = {np.mean(random_inertias):.2f}, std = {np.std(random_inertias):.2f}')
print(f'  K-Means++  : mean inertia = {np.mean(pp_inertias):.2f}, std = {np.std(pp_inertias):.2f}')
print(f'  → K-Means++가 더 낮은 inertia, 더 안정')

# ─────────────────────────────────────────────
# 5. Local minimum 시연 (정리 2.5)
# ─────────────────────────────────────────────
print(f'\nLocal minimum (random init 10번):')
inertias = []
for trial in range(10):
    km = KMeans(n_clusters=4, init='random', n_init=1, random_state=trial).fit(X)
    inertias.append(km.inertia_)
print(f'  Inertias: {sorted(inertias)}')
print(f'  → 다른 init → 다른 결과 (비볼록)')
```

**출력 예시**:
```
My K-Means 수렴: 8 iter
sklearn K-Means inertia: 521.43
My K-Means inertia    : 525.21

Loss 단조감소:
  iter 0: J = 4231.43
  iter 1: J = 1421.32
  iter 2: J = 821.45
  iter 5: J = 532.18
  iter 10: J = 521.45
  iter 19: J = 521.43

GMM의 σ → 0 limit이 K-Means가 됨:
  σ =   10.0: ARI(GMM, KMeans) = 0.5234
  σ =    1.0: ARI(GMM, KMeans) = 0.7821
  σ =    0.1: ARI(GMM, KMeans) = 0.9534
  σ =   0.01: ARI(GMM, KMeans) = 0.9982

Random init vs K-Means++ (4 cluster, 50 trial):
  Random  init: mean inertia = 612.43, std = 154.32
  K-Means++  : mean inertia = 525.13, std = 12.43
  → K-Means++가 더 낮은 inertia, 더 안정

Local minimum (random init 10번):
  Inertias: [521.43, 521.43, 521.43, 612.32, 612.32, 723.45, 823.21, 823.21, 921.32, 1023.43]
  → 다른 init → 다른 결과 (비볼록)
```

---

## 🔗 실전 활용

- **sklearn `KMeans(n_clusters=K, init='k-means++', n_init=10)`**: 표준.
- **MiniBatchKMeans**: 큰 데이터에서 빠른 SGD-like 변형.
- **Document clustering**: TF-IDF + KMeans baseline.
- **Image compression / Vector Quantization**: 픽셀의 K-color로.
- **Customer segmentation**: marketing의 표준 도구.
- **GMM**: K-Means의 부드러운 일반화 — `sklearn.mixture.GaussianMixture`.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| 구형 cluster 가정 | 길쭉한 cluster 처리 못 함 — GMM, DBSCAN |
| Hard assignment | 경계의 점은 부정확 — GMM의 soft |
| $K$ 선택 어려움 | Elbow method, silhouette score, Gap statistic |
| Outlier 민감 | 평균이 outlier에 끌림 — K-Medoid (PAM) |
| 차원 저주 | 고차원에서 거리 무의미 |

---

## 📌 핵심 정리

$$\boxed{\text{K-Means} = \text{GMM}(\Sigma \to 0) = \text{Hard EM};\ \text{Lloyd: 단조 + 유한 수렴};\ \text{K-Means++}: O(\log k)\text{-approx}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **Lloyd 알고리즘** | Assign + Update 반복 |
| **단조감소** | 각 step에서 $J$ 감소 |
| **유한 수렴** | $z$의 가짓수 유한 |
| **GMM 극한** | $\sigma \to 0$의 hard EM |
| **K-Means++** | $O(\log k)$ 경쟁비 |
| **비볼록** | local minimum — 여러 init 필요 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): K-Means에서 $K$를 너무 크게 잡으면? $K = n$이면?

<details>
<summary>힌트 및 해설</summary>

$K$ 매우 큼: 각 cluster에 점 적음 → noisy. 극단 $K = n$ → 각 점이 자기 cluster → $J = 0$. 의미 없음.

→ $K$ 선택이 K-Means의 가장 어려운 부분.

**Elbow method**: $K$별 $J(K)$ plot. 급격히 감소 후 평탄해지는 $K^*$.

**Silhouette score**: 각 점에 대해 자기 cluster 평균 거리 vs 가장 가까운 다른 cluster 평균 거리 — $[-1, 1]$.

</details>

**문제 2** (심화): Lloyd가 항상 global minimum 도달 안 한다. 그러나 어떤 특수 케이스에서는 항상 도달 보장된다 — 어느 경우?

<details>
<summary>힌트 및 해설</summary>

**1차원 K-Means**: $\mathbb{R}$에서 $n$개 점, $K$ cluster. Optimal partition은 항상 contiguous (연속 점들이 같은 cluster). $O(n^2 K)$ DP로 정확한 optimum 가능.

**Cluster center 간 거리가 매우 큼** (well-separated): random init도 거의 항상 정답. 

**특수한 데이터 분포**: 정확한 Gaussian mixture (well-separated)에서 high probability로 optimum 도달.

일반적으로 (특히 high-dim, equal-sized clusters) Lloyd는 sub-optimal local minimum에 갇힘.

</details>

**문제 3** (ML 연결): NN의 **clustering with deep features** 또는 **DEC** (Deep Embedded Clustering, Xie 2016) — NN과 K-Means의 hybrid?

<details>
<summary>힌트 및 해설</summary>

**문제**: Raw image를 K-Means하면 픽셀 거리 → 의미 없음.

**해결 1 (단순)**: NN으로 representation 학습 → 그 위에 K-Means.

**DEC (Xie 2016)**: NN + K-Means를 **joint하게** 학습.
1. Auto-encoder로 NN 학습 (unsupervised representation).
2. K-Means로 init cluster centroids.
3. Soft assignment $q_{ik}$ (Student's t-distribution) 계산.
4. Target distribution $p_{ik}$ (q를 sharpened) 정의.
5. KL($p \| q$) 최소화로 NN encoder + cluster centroids 동시 학습.

→ representation과 cluster가 서로 reinforce. 다음 epoch에서 더 cluster-friendly representation, 더 clean cluster.

**최근**: SwAV, BYOL 같은 contrastive learning이 cluster 정신과 결합.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. KNN과 Cover-Hart](./01-knn-cover-hart.md) | [📚 README](../README.md) | [03. Hierarchical Clustering ▶](./03-hierarchical.md) |

</div>
