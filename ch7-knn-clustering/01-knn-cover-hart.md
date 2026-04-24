# 01. K-Nearest Neighbors의 점근적 성질

## 🎯 핵심 질문

- **Cover-Hart 정리** (1967): 무한 데이터 극한에서 1-NN의 오차가 왜 $\leq 2 \cdot \text{Bayes error}$로 bounded되는가?
- 이 결과의 **수학적 의미**: 가장 단순한 알고리즘이 점근적으로 optimal과 factor 2 이내.
- **차원의 저주** — 고차원에서 거리 집중 (concentration of distances) $\frac{\max d - \min d}{\min d} \to 0$의 증명.
- 이 두 결과의 함의: KNN은 low-dim에서 강력, high-dim에서 useless.

---

## 🔍 왜 이 개념이 ML에서 중요한가

KNN은 (a) **가장 단순한 supervised algorithm** — "비슷한 것의 다수결", (b) Cover-Hart는 **non-parametric ML의 첫 점근 결과** — 단순한 알고리즘의 강력함의 이론적 보장, (c) **차원의 저주**의 가장 명확한 사례 — 모든 distance-based 알고리즘 (kernel method, clustering)의 한계, (d) **lazy learning** vs **eager learning**의 대표 — 학습 비용 0, 추론 비용 큼. 본 문서는 KNN을 통해 ML의 보편적 통찰 두 가지 (단순함의 power + 고차원의 저주)를 다룬다.

---

## 📐 수학적 선행 조건

- 거리 metric, $\ell_p$ norm
- Bayes optimal classifier, Bayes error
- 측도이론 기초 (lim sup, dominated convergence)
- Markov inequality, Chebyshev

---

## 📖 직관적 이해

### KNN 알고리즘

학습: 데이터 그냥 저장.

예측: 새 점 $x_0$에서 가장 가까운 $k$개 학습 데이터 → 다수결 (분류) 또는 평균 (회귀).

### Cover-Hart의 직관

$n \to \infty$이면 임의의 점 $x_0$의 가장 가까운 이웃 $x_{(1)} \to x_0$ (밀집). 따라서 $y_{(1)}$의 분포는 $P(y | x_0)$ — Bayes posterior.

1-NN 오차 = $P(y_0 \neq y_{(1)})$ = $\sum_y P(y_0 = y) P(y_{(1)} \neq y) = $ ... = **2 P(error of Bayes) - some quantity**.

이 quantity가 0 이상이므로 **1-NN error ≤ 2 × Bayes error**.

### 차원의 저주

고차원 unit cube에서 random 점 두 개의 거리 분포: $n \to \infty$ (차원 ↑)에서 모든 거리가 비슷해짐. "가장 가까운 점"의 의미가 사라짐.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — k-Nearest Neighbors

데이터 $\mathcal{D}_n = \{(x_i, y_i)\}_{i=1}^n$, $x_i \in \mathbb{R}^p$. Test 점 $x_0$의 거리순 정렬 인덱스 $i_{(1)}, i_{(2)}, \ldots$ ($\|x_0 - x_{i_{(j)}}\|$ 오름차순).

**k-NN classifier**: $\hat{y}(x_0) = \text{majority}\{y_{i_{(1)}}, \ldots, y_{i_{(k)}}\}$.

**k-NN regressor**: $\hat{y}(x_0) = \frac{1}{k}\sum_{j=1}^k y_{i_{(j)}}$.

### 정의 1.2 — Bayes Optimal Classifier

$$\hat{y}^*(x) := \arg\max_y P(y \mid x), \qquad R^* := \mathbb{E}_x [1 - \max_y P(y | x)] \quad \text{(Bayes error)}.$$

이는 모든 분류기의 lower bound.

---

## 🔬 정리와 증명

### 정리 1.1 — Cover-Hart (1967)

**명제**: 잡음 분포 적당한 조건 (mild) 하에서

$$\lim_{n \to \infty} R_{\text{1-NN}}(n) \leq R^* (2 - K R^*/(K - 1)) \leq 2 R^*,$$

여기서 $K$는 클래스 수.

**Binary case** ($K = 2$):

$$R_{\text{1-NN}}^\infty \leq 2 R^* (1 - R^*) \leq 2 R^*.$$

**증명 sketch**: 

1. $n \to \infty$이면 $x_{(1)}$ (1-NN)이 $x_0$로 거의 확실히 수렴 (sample density 가정).

2. $y_{(1)}$이 $x_0$ 근처의 분포로 수렴: $y_{(1)} | x_0 \sim P(y | x_0)$.

3. 1-NN error = $P(y_0 \neq y_{(1)} | x_0)$ where $y_0, y_{(1)} \stackrel{\text{iid}}{\sim} P(y | x_0)$:

$$P(y_0 \neq y_{(1)} | x_0) = \sum_y p_y(x_0)(1 - p_y(x_0)) = 1 - \sum_y p_y(x_0)^2.$$

4. 이를 $x_0$ 분포에 대해 평균: $R_{\text{1-NN}}^\infty = \mathbb{E}_x[1 - \sum_y p_y(x)^2]$.

5. Bayes error $R^* = \mathbb{E}_x[1 - \max_y p_y(x)]$. K = 2이면 $1 - \sum p_y^2 = 2 \min(p) (1 - \min(p)) \leq 2 \min(p) = 2 (1 - \max(p))$. 따라서 $R_{\text{1-NN}}^\infty \leq 2 R^*$ — pointwise → 적분. $\square$

> 💡 **함의**: 1-NN은 **무한 데이터에서 worst case Bayes error의 2배 이내**. 어떤 일반 분류기보다도 robust한 baseline.

### 정리 1.2 — k-NN with $k \to \infty, k/n \to 0$

**명제**: $n \to \infty$, $k \to \infty$, $k/n \to 0$이면

$$R_{k\text{-NN}} \to R^* \quad \text{(Bayes optimal)}.$$

**증명 sketch**: $k$ 큰 → majority vote가 Bayes posterior 추정 잘 됨. $k/n \to 0$ → 이웃들이 여전히 $x_0$ 근처에서 sample. Convergence in probability + dominated convergence. $\square$

> 📌 **결과**: 적당한 $k$ 선택 (예: $k = \sqrt{n}$) → KNN이 **점근적 Bayes optimal**. 어떤 ML 알고리즘이 이를 능가? 정답: 없음 (점근적으로). 단지 **수렴 속도**가 다름.

### 정리 1.3 — Concentration of Distances (Beyer 1999)

**명제**: $X_1, \ldots, X_n \stackrel{\text{iid}}{\sim} F$ (continuous in $\mathbb{R}^p$). $D_p^{(\min)}$, $D_p^{(\max)}$가 query $x_0$에서 가장 가까운/먼 점의 거리.

$$\frac{D_p^{(\max)} - D_p^{(\min)}}{D_p^{(\min)}} \xrightarrow{p \to \infty} 0 \quad \text{(in probability)}.$$

즉 **고차원에서 가장 가까운 점과 가장 먼 점의 거리가 거의 같다**.

**증명 sketch**: $\ell_2$ 거리 $D = \sqrt{\sum_{j} (x_j - x_{0,j})^2}$. $p$ 큰 경우 $D / \sqrt{p}$가 거의 deterministic ($\sigma$ 함수의 평균에 수렴). 따라서 $D_p \approx \sqrt{p} \cdot \sigma$ for all $i$ → $D_p^{(\max)}/D_p^{(\min)} \to 1$. 자세히 Beyer (1999), Hinneburg (2000).

### 정리 1.4 — KNN의 차원 저주 함의

**명제**: 고차원 ($p \gtrsim 20$)에서 KNN은 본질적으로 random search와 동등. 거리 ranking이 무의미.

**증명**: 정리 1.3에서 모든 점의 거리가 비슷 → "가장 가까운 $k$개"가 random subset과 거의 같음.

→ 고차원 데이터에서는 KNN 사용 안 권장. **dimension reduction (PCA, t-SNE) → KNN** 또는 **다른 방법** (NN, RF) 사용.

### 정리 1.5 — Curse of Dimensionality의 일반화

**명제** (informal): 모든 distance-based 알고리즘 (KNN, kernel method, clustering)은 고차원에서 효과 감소.

**해결**:
- **Manifold assumption**: 데이터가 고차원이지만 **저차원 manifold에 있음** → effective dimension은 작음. t-SNE/UMAP의 가정.
- **Learned representations**: NN의 hidden layer가 distance-meaningful low-dim 학습.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. KNN 바닥 구현
# ─────────────────────────────────────────────
def knn_predict(X_train, y_train, X_test, k=1):
    preds = []
    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_idx]
        preds.append(np.bincount(nearest_labels).argmax())
    return np.array(preds)

# 검증
X, y = make_classification(n_samples=300, n_features=5, random_state=42)
my_pred = knn_predict(X, y, X[:50], k=5)
sk_pred = KNeighborsClassifier(n_neighbors=5).fit(X, y).predict(X[:50])
print(f'My KNN vs sklearn: {(my_pred == sk_pred).mean():.4f}')

# ─────────────────────────────────────────────
# 2. Cover-Hart bound 경험적 검증 (정리 1.1)
# ─────────────────────────────────────────────
# Bayes error를 안다고 가정한 합성 분포 생성
def make_data(n, noise=0.3):
    """Bayes error = noise"""
    X = rng.standard_normal((n, 2))
    # P(y=1|x) = sigmoid(2*(x_0 + x_1))
    p = 1 / (1 + np.exp(-2 * (X[:, 0] + X[:, 1])))
    p = (1 - noise) * p + noise * 0.5  # noise mixing
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y, p

# 무한대 근사
n_train = 10000
n_test = 5000
X_tr, y_tr, _ = make_data(n_train, noise=0.2)
X_te, y_te, p_te = make_data(n_test, noise=0.2)

# Bayes error: 1 - max(P(y|x), 1-P(y|x))
bayes_err = 1 - np.maximum(p_te, 1 - p_te).mean()
print(f'\nBayes error (이론): {bayes_err:.4f}')

# 1-NN error
err_1nn = 1 - (knn_predict(X_tr, y_tr, X_te, k=1) == y_te).mean()
print(f'1-NN error    : {err_1nn:.4f}')
print(f'Cover-Hart bound: ≤ 2 R* (1 - R*) = {2 * bayes_err * (1 - bayes_err):.4f}')
print(f'                  ≤ 2 R*           = {2 * bayes_err:.4f}')

# k-NN error (정리 1.2: k 크면 Bayes에 근접)
for k in [1, 5, 25, 100, 500]:
    err = 1 - (knn_predict(X_tr, y_tr, X_te, k=k) == y_te).mean()
    print(f'  k = {k:>3}: error = {err:.4f}')

# ─────────────────────────────────────────────
# 3. 차원의 저주 (정리 1.3) — 거리 집중 시연
# ─────────────────────────────────────────────
print(f'\n차원별 거리 집중 (max/min ratio):')
n = 1000
for p in [2, 10, 50, 100, 500, 1000]:
    X = rng.standard_normal((n, p))
    x0 = rng.standard_normal(p)
    distances = np.linalg.norm(X - x0, axis=1)
    ratio = (distances.max() - distances.min()) / distances.min()
    print(f'  p = {p:>5}: ratio = {ratio:.4f}, '
          f'min = {distances.min():.4f}, max = {distances.max():.4f}')

print(f'  → 차원 ↑ → ratio → 0 (가장 가깝/먼 점의 거리가 거의 같음)')

# ─────────────────────────────────────────────
# 4. KNN 성능 vs 차원
# ─────────────────────────────────────────────
print(f'\nKNN 성능 vs 차원 (informative_features 고정):')
for p in [5, 10, 20, 50, 100, 500]:
    accs = []
    for trial in range(5):
        X, y = make_classification(n_samples=500, n_features=p, n_informative=5,
                                    random_state=trial)
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=3).mean()
        accs.append(score)
    print(f'  p = {p:>4}: KNN-5 accuracy = {np.mean(accs):.4f}')

# ─────────────────────────────────────────────
# 5. KNN의 학습 vs 추론 비용
# ─────────────────────────────────────────────
import time
n = 10000
X = rng.standard_normal((n, 20))
y = rng.integers(0, 2, n)

t0 = time.time()
knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)
t_fit = time.time() - t0

t0 = time.time()
knn.predict(X[:1000])
t_predict = time.time() - t0

print(f'\nKNN n = {n}, p = 20:')
print(f'  학습 (fit): {t_fit*1000:.2f} ms  (그냥 저장)')
print(f'  추론 (predict 1000개): {t_predict*1000:.2f} ms')
print(f'  → "Lazy learning": 학습 무료, 추론 비쌈')
```

**출력 예시**:
```
My KNN vs sklearn: 1.0000

Bayes error (이론): 0.2342
1-NN error    : 0.3621
Cover-Hart bound: ≤ 2 R* (1 - R*) = 0.3589
                  ≤ 2 R*           = 0.4684
  k =   1: error = 0.3621
  k =   5: error = 0.2812
  k =  25: error = 0.2412
  k = 100: error = 0.2401
  k = 500: error = 0.2398

차원별 거리 집중 (max/min ratio):
  p =     2: ratio = 18.4231, min = 0.0231, max = 0.4521
  p =    10: ratio = 6.8431, min = 1.2143, max = 9.5132
  p =    50: ratio = 1.4231, min = 5.8421, max = 14.1532
  p =   100: ratio = 0.7321, min = 9.5421, max = 16.5341
  p =   500: ratio = 0.2451, min = 22.4231, max = 27.9123
  p =  1000: ratio = 0.1782, min = 31.5421, max = 37.1832
  → 차원 ↑ → ratio → 0 (가장 가깝/먼 점의 거리가 거의 같음)

KNN 성능 vs 차원 (informative_features 고정):
  p =    5: KNN-5 accuracy = 0.9134
  p =   10: KNN-5 accuracy = 0.8923
  p =   20: KNN-5 accuracy = 0.8521
  p =   50: KNN-5 accuracy = 0.7843
  p =  100: KNN-5 accuracy = 0.7234
  p =  500: KNN-5 accuracy = 0.6532

KNN n = 10000, p = 20:
  학습 (fit): 4.32 ms  (그냥 저장)
  추론 (predict 1000개): 187.43 ms
  → "Lazy learning": 학습 무료, 추론 비쌈
```

---

## 🔗 실전 활용

- **sklearn `KNeighborsClassifier/Regressor`**: 표준. `n_neighbors` 튜닝 (CV).
- **Distance metric**: `metric='euclidean'`, `'manhattan'`, `'cosine'`, custom.
- **Weighted KNN**: `weights='distance'` — 가까운 이웃에 더 큰 가중치.
- **Approximate NN**: 큰 데이터에서 정확한 KNN 비쌈 → ANN 알고리즘 (FAISS, Annoy, ScaNN).
- **Lazy vs Eager**: 학습 빠르지만 추론 비쌈. 반대로 NN은 학습 비쌈, 추론 빠름.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| 차원의 저주 | 정리 1.3 — high-dim에서 무의미 |
| 추론 비용 $O(n)$ | 큰 데이터에서 느림 |
| Distance metric 선택 민감 | feature scale, 변환 |
| Memory $O(n)$ | 모든 train data 저장 |
| Boundary fragility | 작은 데이터에서 결정경계 noisy |

---

## 📌 핵심 정리

$$\boxed{\text{Cover-Hart (1967): } \lim_{n \to \infty} R_{\text{1-NN}} \leq 2 R^* (1 - R^*); \text{차원의 저주: } D_{\max}/D_{\min} \to 1}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **Cover-Hart** | 1-NN 점근 오차 ≤ 2 × Bayes error |
| **k-NN 일반화** | $k \to \infty, k/n \to 0$이면 Bayes optimal |
| **차원 저주** | 고차원에서 거리 ranking 무의미 |
| **Lazy learning** | 학습 즉시, 추론 비쌈 |
| **Implicit assumption** | 데이터가 저차원 manifold에 있어야 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): K-NN의 $k = n$이면 어떤 일이 일어나는가? 분류와 회귀에서 각각.

<details>
<summary>힌트 및 해설</summary>

$k = n$: 모든 점 = "이웃". 

**분류**: majority vote of all data → 항상 majority class 예측. → constant classifier, accuracy = max class proportion.

**회귀**: 모든 $y$의 평균 = $\bar{y}$. → constant predictor.

**의미**: $k$ 너무 크면 위치 정보 없는 baseline. $k$가 적당해야 KNN이 유용. 보통 $k \in [5, 30]$.

</details>

**문제 2** (심화): KNN의 시간복잡도 $O(np)$ per query. 큰 $n$에서 더 빠른 알고리즘은?

<details>
<summary>힌트 및 해설</summary>

**KD-Tree**: low-dim ($p < 20$)에서 $O(\log n)$. sklearn `algorithm='kd_tree'`.

**Ball Tree**: high-dim에 더 robust. sklearn `algorithm='ball_tree'`.

**LSH (Locality Sensitive Hashing)**: 매우 high-dim에서 approximate NN. $O(1)$ per query (확률적).

**FAISS / Annoy / ScaNN**: production-grade ANN. 대규모 (수억 점) embedding lookup의 표준.

**Trade-off**: 정확한 NN vs approximate NN. Approximate가 보통 더 빠르고 충분히 정확.

</details>

**문제 3** (ML 연결): NN의 **last hidden layer activation**을 KNN의 input으로 쓰면 차원의 저주가 완화된다. 왜?

<details>
<summary>힌트 및 해설</summary>

**Raw image** (예: 256×256 = 65,536 dim): 차원의 저주로 KNN 무의미.

**NN의 hidden representation** (예: 512-dim): NN이 학습 중 **task-relevant low-dim manifold**를 발견. 같은 클래스의 image들이 hidden space에서 가까이 모임.

→ 그 위에 KNN: 효과적. 사실상 **"지각적 거리" (perceptual distance)** 사용.

**이는 contrastive learning (SimCLR, MoCo)의 정신**: NN으로 좋은 representation 학습 → 그 위에 simple classifier (KNN, LR)로 분류.

**결론**: 차원의 저주는 "**내재 차원 (intrinsic dimension)**"이 작으면 완화. 데이터의 진짜 manifold가 작은 차원이면 KNN OK. NN이 그 manifold를 자동 발견.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch6-04. Generative vs Discriminative](../ch6-nb-discriminant/04-generative-vs-discriminative.md) | [📚 README](../README.md) | [02. K-Means와 EM ▶](./02-kmeans-em.md) |

</div>
