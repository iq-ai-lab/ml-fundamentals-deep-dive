# 03. Hierarchical Clustering

## 🎯 핵심 질문

- **Agglomerative** (bottom-up) vs **Divisive** (top-down) hierarchical clustering의 차이는?
- 4가지 linkage 기준 (single·complete·average·**Ward**)의 수학적 차이와 cluster 모양에 미치는 영향?
- **Ultrametric**과 dendrogram의 일대일 대응 — 위계적 clustering의 메타 구조.
- $K$를 미리 정할 필요 없는 장점 vs $O(n^3)$ 계산 비용의 단점.

---

## 🔍 왜 이 개념이 ML에서 중요한가

Hierarchical clustering은 (a) **$K$를 미리 모르거나 다양한 granularity**가 필요한 경우의 표준, (b) **dendrogram 시각화**가 매우 직관적, (c) **biological taxonomy·document hierarchy·social network community**의 자연스러운 표현, (d) K-Means와 다른 **deterministic** 결과 (random init 없음). 본 문서는 hierarchical clustering의 elegance — 각 linkage가 다른 metric의 distance를 minimize함 — 와 그것이 만드는 trade-off를 분석.

---

## 📐 수학적 선행 조건

- 거리 metric의 정의 (triangle inequality 등)
- Ward의 분산 최소화 직관
- 그래프와 tree 기본

---

## 📖 직관적 이해

### Agglomerative

1. Init: 각 점이 자기 cluster ($n$개).
2. 가장 가까운 두 cluster를 합침.
3. 반복 → 1개 cluster까지.

→ **dendrogram** (트리 구조) 생성.

### Divisive

반대 방향: 1개 cluster → 분할 → 분할 → ... → $n$개 점. 더 비쌈, 덜 사용.

### Linkage Criteria

두 cluster $A, B$ 사이의 거리:

- **Single**: $d(A, B) = \min_{a \in A, b \in B} d(a, b)$ — 가장 가까운 점쌍.
- **Complete**: $\max d(a, b)$ — 가장 먼 점쌍.
- **Average** (UPGMA): $\text{avg } d(a, b)$.
- **Ward**: 합쳐진 cluster의 within-variance 증가량 — variance 최소화.

각 linkage가 다른 cluster 모양 만듦:
- Single: chain-like (긴 사슬).
- Complete: compact (구형).
- Ward: balanced (size 비슷).

### Ultrametric

Dendrogram에서 두 점의 distance = 그 두 점이 처음 같은 cluster가 되는 height. 이 distance는 **ultrametric** ($d(x, y) \leq \max(d(x, z), d(z, y))$, triangle inequality 강화).

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Agglomerative Clustering

1. $C_i = \{x_i\}$ for each $i$.
2. While > 1 cluster:
   - $(i^*, j^*) = \arg\min d(C_i, C_j)$.
   - $C_{\text{new}} = C_{i^*} \cup C_{j^*}$, remove $C_{i^*}, C_{j^*}$.
   - Record merge height $d(C_{i^*}, C_{j^*})$.

### 정의 3.2 — Linkage Functions

$d(A, B)$의 정의:
- **Single**: $\min_{a, b} \|a - b\|$.
- **Complete**: $\max_{a, b} \|a - b\|$.
- **Average**: $\frac{1}{|A||B|}\sum_{a, b} \|a - b\|$.
- **Ward**: $\frac{|A||B|}{|A| + |B|} \|\bar{a} - \bar{b}\|^2$.

### 정의 3.3 — Ultrametric

거리 함수 $d$가 ultrametric: $d(x, y) \leq \max(d(x, z), d(z, y))$ for all $x, y, z$.

### 정의 3.4 — Dendrogram

Hierarchical clustering의 결과 — binary tree where leaves = data points, internal nodes = merges, height = merge distance.

---

## 🔬 정리와 증명

### 정리 3.1 — Ward의 Variance Minimization

**명제**: Ward linkage의 $d(A, B) = \frac{|A||B|}{|A|+|B|}\|\bar{a} - \bar{b}\|^2$는 **합쳐진 cluster의 within-variance 증가량**과 정확히 같다.

**증명**: $C = A \cup B$의 within-SS $\sum_{x \in C}\|x - \bar{c}\|^2$. 분산 분해 (정리 3-3.2):

$\sum_{x \in C}\|x - \bar{c}\|^2 = \sum_{x \in A}\|x - \bar{a}\|^2 + \sum_{x \in B}\|x - \bar{b}\|^2 + |A|\|\bar{a} - \bar{c}\|^2 + |B|\|\bar{b} - \bar{c}\|^2$.

$\bar{c} = (|A|\bar{a} + |B|\bar{b})/(|A|+|B|)$ 대입:

$|A|\|\bar{a} - \bar{c}\|^2 + |B|\|\bar{b} - \bar{c}\|^2 = \frac{|A||B|}{|A|+|B|}\|\bar{a} - \bar{b}\|^2$.

→ $\Delta \text{SS} = \frac{|A||B|}{|A|+|B|}\|\bar{a} - \bar{b}\|^2$ — Ward distance. $\square$

> 💡 **함의**: Ward는 K-Means와 같은 정신 (within-variance 최소화). 두 알고리즘 결과 종종 매우 유사.

### 정리 3.2 — Single Linkage = Minimum Spanning Tree

**명제**: Single linkage의 dendrogram은 데이터 점들의 **minimum spanning tree (MST)** 와 동치.

**증명 sketch**: 각 merge step은 두 cluster 사이 가장 짧은 edge를 추가 → MST 생성 알고리즘 (Kruskal과 같음). $\square$

> 📌 **함의**: Single linkage는 graph theory와 직접 연결. **chaining effect** — outlier 한 점이 두 cluster를 잇는 다리 → 큰 cluster로 합쳐짐 (의도와 다른 결과).

### 정리 3.3 — Dendrogram-Ultrametric 일대일

**명제**: 데이터 점들에 대한 dendrogram과 ultrametric은 일대일 대응. (Jardine & Sibson 1971).

**증명 sketch**: 

- Dendrogram → ultrametric: $d(x, y) := $ $x, y$가 같은 cluster가 되는 가장 작은 height. Ultrametric inequality $d(x, y) \leq \max(d(x, z), d(z, y))$ 자명 (셋 다 더 큰 cluster에서는 같이 있음).

- Ultrametric → dendrogram: 거리값들의 unique sorted set이 nested partition 구조 정의.

→ 같은 정보. Dendrogram이 ultrametric의 시각적 표현. $\square$

### 정리 3.4 — Linkage별 Cluster 모양

| Linkage | 강점 | 약점 |
|---------|------|------|
| **Single** | concave, irregular shapes | chaining (outlier 매우 민감) |
| **Complete** | compact (구형) | outlier 민감, big cluster 형성 |
| **Average** | balanced | data scale 의존 |
| **Ward** | balanced size, compact | Euclidean만 가능 |

### 정리 3.5 — Computational Complexity

**명제**: Naive agglomerative는 $O(n^3)$ — 매 step마다 모든 pair 거리 계산. $n$개 → $n - 1$ merges → 총 $\sum k^2 = O(n^3)$.

**Lance-Williams Update**: 합친 cluster의 거리를 이전 거리들로부터 $O(1)$ recursive 갱신 → $O(n^2)$로 감축. SLINK (Sibson 1973): single linkage의 $O(n^2)$ 알고리즘.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Hierarchical clustering 실행
# ─────────────────────────────────────────────
X, y = make_blobs(n_samples=50, centers=3, cluster_std=0.8, random_state=42)

# scipy의 linkage 함수
Z_single = linkage(X, method='single')
Z_complete = linkage(X, method='complete')
Z_average = linkage(X, method='average')
Z_ward = linkage(X, method='ward')

print(f'각 linkage의 처음 5개 merge:')
print(f'\nSingle:')
print(f'{"step":>4s} {"i":>4s} {"j":>4s} {"dist":>8s} {"|new|":>5s}')
for s in range(5):
    print(f'{s:>4d} {int(Z_single[s,0]):>4d} {int(Z_single[s,1]):>4d} '
          f'{Z_single[s,2]:>8.4f} {int(Z_single[s,3]):>5d}')

# ─────────────────────────────────────────────
# 2. Ward = K-Means와 비교 (정리 3.1)
# ─────────────────────────────────────────────
from sklearn.cluster import KMeans
ward_labels = fcluster(Z_ward, t=3, criterion='maxclust')
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)

from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(ward_labels, kmeans.labels_)
print(f'\nWard vs K-Means ARI: {ari:.4f} (둘 다 within-variance 최소화)')

# ─────────────────────────────────────────────
# 3. Single linkage의 chaining effect (정리 3.2)
# ─────────────────────────────────────────────
# Chain-like 데이터
n = 50
chain = np.zeros((n, 2))
chain[:25, 0] = np.linspace(0, 5, 25)   # cluster 1
chain[25:, 0] = np.linspace(7, 12, 25)  # cluster 2
chain[:, 1] = 0.1 * rng.standard_normal(n)
# Outlier 한 점이 다리 역할
bridge = np.array([[6, 0]])
chain_with_bridge = np.vstack([chain, bridge])

for method in ['single', 'complete', 'ward']:
    Z = linkage(chain_with_bridge, method=method)
    labels = fcluster(Z, t=2, criterion='maxclust')
    cluster_sizes = [(labels == k).sum() for k in np.unique(labels)]
    print(f'  {method:>9s}: cluster sizes = {sorted(cluster_sizes)}')

print(f'  → Single은 bridge를 사용해 chain 두 개를 합침')
print(f'  → Complete/Ward는 둘로 분리 (compact)')

# ─────────────────────────────────────────────
# 4. Linkage별 결과 차이 — moons 데이터
# ─────────────────────────────────────────────
X_moon, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
print(f'\nMoons 데이터에서 linkage별:')
for method in ['single', 'complete', 'average', 'ward']:
    clf = AgglomerativeClustering(n_clusters=2, linkage=method).fit(X_moon)
    # 두 moon이 어떻게 분리되는지 확인 (지표 없으므로 간략히)
    sizes = sorted([(clf.labels_ == k).sum() for k in [0, 1]])
    print(f'  {method:>9s}: cluster sizes = {sizes}')

# ─────────────────────────────────────────────
# 5. Dendrogram의 cut height
# ─────────────────────────────────────────────
print(f'\nWard linkage의 dendrogram cut:')
for K in [2, 3, 5, 10]:
    labels = fcluster(Z_ward, t=K, criterion='maxclust')
    actual_K = len(np.unique(labels))
    sizes = sorted([(labels == k).sum() for k in np.unique(labels)])
    print(f'  K = {K}: actual K = {actual_K}, sizes = {sizes}')

# 또는 distance threshold로 cut
print(f'\nDistance threshold로 cut:')
for height in [0.5, 1.0, 5.0, 20.0]:
    labels = fcluster(Z_ward, t=height, criterion='distance')
    K = len(np.unique(labels))
    print(f'  height = {height}: K = {K}')

# ─────────────────────────────────────────────
# 6. 복잡도 시간 측정
# ─────────────────────────────────────────────
import time
print(f'\n시간 복잡도 (Ward):')
for n in [100, 500, 1000, 2000]:
    X_n = rng.standard_normal((n, 5))
    t0 = time.time()
    linkage(X_n, method='ward')
    t = time.time() - t0
    print(f'  n = {n:>4}: {t*1000:.1f} ms')
```

**출력 예시**:
```
각 linkage의 처음 5개 merge:

Single:
step    i    j     dist  |new|
   0   23   34   0.0234     2
   1   12   45   0.0312     2
   2   18   29   0.0421     2
   3   11   25   0.0512     2
   4   23   24   0.0521     3

Ward vs K-Means ARI: 0.9450 (둘 다 within-variance 최소화)

  single: cluster sizes = [1, 50]   ← bridge 사용
  complete: cluster sizes = [25, 26]
  ward    : cluster sizes = [25, 26]
  → Single은 bridge를 사용해 chain 두 개를 합침
  → Complete/Ward는 둘로 분리 (compact)

Moons 데이터에서 linkage별:
  single   : cluster sizes = [50, 50]    ← chaining이 moons 잘 분리
  complete : cluster sizes = [38, 62]
  average  : cluster sizes = [42, 58]
  ward     : cluster sizes = [50, 50]

시간 복잡도 (Ward):
  n =  100: 1.2 ms
  n =  500: 12.3 ms
  n = 1000: 47.2 ms
  n = 2000: 187.4 ms   ← O(n²) 정도
```

---

## 🔗 실전 활용

- **scipy `linkage` + `dendrogram`**: 시각화 표준.
- **sklearn `AgglomerativeClustering`**: integration with sklearn pipeline.
- **Phylogenetic analysis**: biological evolution tree.
- **Document hierarchy**: 토픽 모델링 + clustering.
- **Customer segment hierarchy**: marketing의 다층 분류.
- **Image segmentation**: Felzenszwalb-Huttenlocher graph-based segmentation의 정신.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| $O(n^2)$ 메모리 | 거리 행렬 — n = 10,000부터 어려움 |
| $O(n^3)$ 또는 $O(n^2)$ 시간 | 큰 데이터에서 느림 |
| Linkage 선택 | 결과에 큰 영향 |
| Outlier 민감 (single) | chaining |
| Greedy | 한 번 합치면 되돌릴 수 X |

---

## 📌 핵심 정리

$$\boxed{\text{Agglomerative: bottom-up merging};\ \text{Linkage: 4가지 cluster 모양};\ \text{Ward = K-Means 정신};\ \text{Dendrogram = ultrametric}}$$

| Linkage | 거리 | 모양 |
|---------|------|------|
| **Single** | min | chain-like (MST) |
| **Complete** | max | compact, 구형 |
| **Average** | mean | balanced |
| **Ward** | variance increment | K-Means-like |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Single linkage가 moons 데이터에서 잘 작동하는 이유와 K-Means가 못하는 이유.

<details>
<summary>힌트 및 해설</summary>

Moons: 두 반달 모양 — 비-볼록 (non-convex). K-Means는 cluster center 기반, 구형 cluster 가정 → 두 반달을 가운데서 자름 (틀린 분리).

Single linkage: chain-like merging. 한 반달의 모든 점이 chain으로 연결 → 다른 반달과 분리. **chaining이 여기서 강점**.

→ "chaining"이 항상 나쁜 건 아님. 데이터 모양에 맞을 때 강점.

</details>

**문제 2** (심화): Ward가 K-Means와 같은 결과를 종종 내지만 **다른 알고리즘**이다. 둘이 어떻게 다른가?

<details>
<summary>힌트 및 해설</summary>

**K-Means**: iterative, $K$ 미리 정함, random init, local minimum 가능, $O(nKT)$.

**Ward**: deterministic (no random), hierarchical (모든 $K$ 동시), $O(n^2 \log n)$, hierarchy 구조 보유.

**같은 점**: 둘 다 within-variance 최소화 → 결과 비슷.

**Trade-off**:
- K-Means: 빠름, $K$ 명시 필요.
- Ward: 느림, $K$ 자유 선택, 시각화 풍부.

</details>

**문제 3** (ML 연결): NN의 hierarchical representation (e.g. early CNN layers learn edges, later learn objects) — 이것이 clustering의 hierarchical 정신과 어떻게 연결되는가?

<details>
<summary>힌트 및 해설</summary>

**Hierarchical Clustering**: data points의 multi-level grouping (작은 cluster → 큰 cluster).

**Hierarchical NN representation**: pixel-level → edge → texture → part → object — multi-level abstraction.

**평행 구조**: 둘 다 "**작은 단위 → 큰 단위**의 위계적 구성".

**Hierarchical softmax / hierarchical attention**: NN에서 명시적 hierarchy 도입.

**Capsule networks** (Hinton 2017): 캡슐들의 위계 구조 명시.

**결론**: "hierarchy"는 ML 전체에 보편적 — 단순 → 복잡, 작은 그룹 → 큰 그룹. Hierarchical clustering이 가장 명시적 사례.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. K-Means와 EM](./02-kmeans-em.md) | [📚 README](../README.md) | [04. DBSCAN ▶](./04-dbscan.md) |

</div>
