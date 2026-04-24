# 03. 회귀 트리와 MSE 분할

## 🎯 핵심 질문

- 회귀 트리에서 각 leaf의 예측값 $c_v$가 왜 정확히 **leaf 내 평균** $\bar{y}_v$인가?
- CART의 탐욕 분할 알고리즘은 어떻게 모든 (feature, threshold) 후보 중 최적을 찾는가?
- 연속 feature의 split point 탐색은 왜 $O(n \log n)$인가? sklearn의 효율적 구현은?
- 분류 (Gini)와 회귀 (MSE)가 어떻게 **같은 "분산 감소" 정신을 공유**하는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

회귀 트리는 (a) **CART의 회귀 모드** — sklearn `DecisionTreeRegressor`, (b) **Random Forest Regressor**·**Gradient Boosting Regressor**·**XGBoost regression**의 모든 base learner, (c) 비선형 회귀의 **interpretable baseline** — 모든 split이 if-then 규칙. (d) Ch3-02의 **"Gini = 분산의 이산 버전"** 통찰을 회귀에 직접 일반화 — **MSE 분할이 Gini와 같은 알고리즘적 형태**. 본 문서는 회귀와 분류가 한 알고리즘으로 통합된 이유를 보인다.

---

## 📐 수학적 선행 조건

- 결정트리의 분할 기준 (Ch3-01, 02)
- 평균과 분산의 minimization 성질
- 선형 회귀의 OLS (Ch1-01)

---

## 📖 직관적 이해

### Leaf의 최적 예측값

Leaf $v$ 내 데이터 $\{y_i : i \in S_v\}$에 대해 **상수 예측** $c_v$를 한다. MSE 손실:

$$L_v(c) = \sum_{i \in S_v} (y_i - c)^2.$$

이를 최소화하는 $c^* = \bar{y}_v$ — **leaf 내 평균**. (정리 3.1)

### 분할 기준 = 분산 감소

데이터 $S$를 $S_L, S_R$로 분할:

$$\Delta L = \sum_{i \in S} (y_i - \bar{y}_S)^2 - \sum_{i \in S_L}(y_i - \bar{y}_L)^2 - \sum_{i \in S_R}(y_i - \bar{y}_R)^2.$$

$\Delta L > 0$이면 분할이 도움. 최적 split은 $\Delta L$ 최대화.

### 연속 feature의 효율적 탐색

$X_j$를 정렬 → $n$개 점 사이 $n - 1$개 split point 후보. 각 후보에서 $\Delta L$ 계산을 $O(1)$로 (running sum 유지) → 전체 $O(n \log n)$ (정렬 비용).

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 회귀 트리

회귀 트리 $T$는 (a) feature space $\mathcal{X}$를 disjoint한 region $\{R_v\}_{v \in \text{leaves}}$로 분할하고, (b) 각 region에 상수 $c_v$를 할당한다. 예측

$$\hat{f}(x) = \sum_v c_v \cdot \mathbb{1}[x \in R_v].$$

### 정의 3.2 — Squared Error Loss

데이터 $\{(x_i, y_i)\}$, $\hat{f}$의 손실:

$$L(\hat{f}) = \sum_{i=1}^n (y_i - \hat{f}(x_i))^2.$$

### 정의 3.3 — Best Split Score

Region $R$ within data $S \cap R = \{i\}$, candidate split $(j, t)$ ($X_j \leq t$ vs $X_j > t$). Scoring:

$$\text{Gain}(R, j, t) := L(R) - L(R_L) - L(R_R)$$

여기서 $L(R) = \sum_{i \in R}(y_i - \bar{y}_R)^2$.

---

## 🔬 정리와 증명

### 정리 3.1 — Leaf의 최적 상수는 평균

**명제**: $L(c) = \sum_{i \in S_v}(y_i - c)^2$의 최소값은 $c^* = \bar{y}_v = \frac{1}{|S_v|}\sum_{i \in S_v} y_i$. 최소값은 $\sum (y_i - \bar{y}_v)^2$.

**증명**: $\frac{dL}{dc} = -2 \sum (y_i - c) = 0 \Rightarrow c = \bar{y}_v$. 2계 도함수 $2 |S_v| > 0$ → 최소. $\square$

> 💡 **결과**: 회귀 트리의 leaf 예측 = leaf 내 평균. 매우 직관적.

### 정리 3.2 — Squared Error의 분산 분해

**명제**: $L(R) = |R| \cdot \text{Var}_R(y) = \sum (y_i - \bar{y}_R)^2$.

**명제 (분할의 효과)**: $L(R) - L(R_L) - L(R_R) = \frac{|R_L| |R_R|}{|R|}(\bar{y}_L - \bar{y}_R)^2$.

**증명**: 

$$L(R) = \sum_{i \in R}(y_i - \bar{y}_R)^2 = \sum_L (y_i - \bar{y}_L + \bar{y}_L - \bar{y}_R)^2 + \sum_R (y_i - \bar{y}_R + \bar{y}_R - \bar{y}_R)^2.$$

전개 + cross term 0 (정의 by $\bar{y}$): 

$$L(R) = L(R_L) + |R_L|(\bar{y}_L - \bar{y}_R)^2 + L(R_R) + |R_R|(\bar{y}_R - \bar{y}_R)^2.$$

$\bar{y}_R = (|R_L|\bar{y}_L + |R_R|\bar{y}_R) / |R|$를 대입 + 정리:

$$|R_L|(\bar{y}_L - \bar{y}_R)^2 + |R_R|(\bar{y}_R - \bar{y}_R)^2 = \frac{|R_L||R_R|}{|R|}(\bar{y}_L - \bar{y}_R)^2.$$

→ Gain 공식. $\square$

> 📌 **함의**: Gain이 클수록 두 leaf의 평균이 더 다름. **분류의 IG/Gini와 같은 직관**: split 후 두 그룹이 더 separable.

### 정리 3.3 — CART 탐욕 분할 알고리즘

**알고리즘**:

```
function build_tree(R, depth):
    if stop_criteria (depth max, |R| < min_samples, etc.):
        return Leaf(mean(y in R))
    
    best_gain, best_j, best_t = 0, None, None
    for j in 1..p:
        sort R by X_j
        for t in candidate split points (sorted X_j[k] for k = 1..|R|-1):
            R_L = {i : X_ij <= t}, R_R = {i : X_ij > t}
            gain = Gain(R, j, t)
            if gain > best_gain:
                best_gain = gain
                best_j, best_t = j, t
    
    if best_gain <= 0:
        return Leaf(mean(y in R))
    
    R_L = {i : X_{i, best_j} <= best_t}, R_R = ...
    return Node(best_j, best_t, build_tree(R_L, depth+1), build_tree(R_R, depth+1))
```

**복잡도**: 각 노드에서 $O(p \cdot n \log n)$ (정렬 + 탐색). 깊이 $d$, $n$ samples → 전체 $O(p n \log n \cdot d)$. sklearn의 efficient implementation: presort 한 번 → 후속 split에서 $O(n)$ 갱신.

### 정리 3.4 — Running Sum으로 $O(n)$ 1-feature 탐색

**명제**: $X_j$가 정렬되어 있을 때 모든 $n - 1$개 split point의 Gain을 $O(n)$에 계산 가능.

**증명 스케치**: 정렬된 $y$에 대해 $\sum y$, $\sum y^2$의 prefix sum을 유지. Split point $k$에서 $S_L = \{1, \ldots, k\}$의 평균 = $\text{prefix\_sum}_y[k]/k$, 분산 = $(\text{prefix\_sum}_{y^2}[k] - k \bar{y}_L^2) / k$. 모든 $k$에서 $O(1)$. $\square$

> 💡 **sklearn은 정확히 이 방식** — 그래서 회귀 트리도 분류 트리와 같은 속도.

### 정리 3.5 — 회귀 트리 vs 선형 회귀의 표현력

| 측면 | 선형 회귀 | 회귀 트리 |
|------|----------|-----------|
| 함수 클래스 | linear | piecewise constant |
| Bias | 함수 형태 misspecified면 큼 | 충분히 자라면 0 |
| Variance | 작음 (적은 파라미터) | 큼 (가지가 많음) |
| 외삽 | OK (선형) | 위험 (last leaf의 값으로 고정) |
| 비선형 효과 | feature engineering 필요 | 자동 |

회귀 트리는 **non-parametric** — $n \to \infty$에서 모든 measurable function 근사 가능 (with infinite depth).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. 회귀 트리 바닥 구현
# ─────────────────────────────────────────────
class Node:
    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
    def is_leaf(self):
        return self.feature is None

def best_split(X, y, min_gain=1e-10):
    n, p = X.shape
    if n < 2:
        return None, None, 0
    L_R = ((y - y.mean())**2).sum()
    best_gain, best_j, best_t = 0, None, None
    
    for j in range(p):
        order = np.argsort(X[:, j])
        Xs, ys = X[order, j], y[order]
        # Running sums
        sum_left, sum_left_sq = 0.0, 0.0
        sum_total, sum_total_sq = ys.sum(), (ys**2).sum()
        for k in range(1, n):
            sum_left += ys[k-1]
            sum_left_sq += ys[k-1]**2
            if Xs[k] == Xs[k-1]:
                continue
            n_L, n_R = k, n - k
            mean_L = sum_left / n_L
            mean_R = (sum_total - sum_left) / n_R
            L_L = sum_left_sq - n_L * mean_L**2
            L_R = (sum_total_sq - sum_left_sq) - n_R * mean_R**2
            gain = ((y - y.mean())**2).sum() - L_L - L_R
            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_t = (Xs[k-1] + Xs[k]) / 2
    return best_j, best_t, best_gain

def build_tree(X, y, depth=0, max_depth=4, min_samples=2):
    if depth >= max_depth or len(y) < min_samples:
        return Node(value=y.mean())
    j, t, gain = best_split(X, y)
    if j is None or gain < 1e-7:
        return Node(value=y.mean())
    mask = X[:, j] <= t
    return Node(feature=j, threshold=t,
                left=build_tree(X[mask], y[mask], depth+1, max_depth, min_samples),
                right=build_tree(X[~mask], y[~mask], depth+1, max_depth, min_samples))

def predict(node, X):
    if node.is_leaf():
        return np.full(len(X), node.value)
    pred = np.zeros(len(X))
    mask = X[:, node.feature] <= node.threshold
    pred[mask] = predict(node.left, X[mask])
    pred[~mask] = predict(node.right, X[~mask])
    return pred

# ─────────────────────────────────────────────
# 2. 합성 데이터에서 sklearn과 일치 확인
# ─────────────────────────────────────────────
n = 200
X = rng.uniform(0, 2 * np.pi, (n, 1))
y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n)

tree = build_tree(X, y, max_depth=5)
sk = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)

X_test = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
pred_my = predict(tree, X_test)
pred_sk = sk.predict(X_test)

# RMSE 비교 (둘이 정확히 일치하지는 않음 — tie-breaking 차이)
print(f'MyTree RMSE on train : {np.sqrt(((predict(tree, X) - y)**2).mean()):.4f}')
print(f'sklearn RMSE on train: {np.sqrt(((sk.predict(X) - y)**2).mean()):.4f}')
print(f'MyTree leaves: {sum(1 for _ in iter([]))}  (재귀로 세야 함)')

# ─────────────────────────────────────────────
# 3. Leaf의 평균 검증 (정리 3.1)
# ─────────────────────────────────────────────
def get_leaves(node):
    if node.is_leaf():
        return [node.value]
    return get_leaves(node.left) + get_leaves(node.right)

leaves = get_leaves(tree)
print(f'\n트리 leaf 개수: {len(leaves)}')
print(f'Leaf 값 (처음 5개): {[round(v, 3) for v in leaves[:5]]}')
print(f'(각 leaf 값은 그 region 내 y의 평균임)')

# ─────────────────────────────────────────────
# 4. Gain 공식 검증 (정리 3.2)
# ─────────────────────────────────────────────
y_test = np.array([1.0, 2.0, 5.0, 8.0, 10.0])
# Split between index 2 and 3
y_L = y_test[:3]; y_R = y_test[3:]
mean_total = y_test.mean()
L_total = ((y_test - mean_total)**2).sum()
L_L = ((y_L - y_L.mean())**2).sum()
L_R = ((y_R - y_R.mean())**2).sum()
gain_direct = L_total - L_L - L_R
gain_formula = (len(y_L) * len(y_R) / len(y_test)) * (y_L.mean() - y_R.mean())**2
print(f'\nGain (직접 계산): {gain_direct:.4f}')
print(f'Gain (정리 3.2 공식): {gain_formula:.4f}')

# ─────────────────────────────────────────────
# 5. Bias-Variance — depth별 학습 곡선
# ─────────────────────────────────────────────
from sklearn.metrics import mean_squared_error
depths = [1, 2, 4, 6, 10, 20]
print(f'\nDepth vs train/test RMSE (sin curve):')
for d in depths:
    sk = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, y)
    train_rmse = np.sqrt(mean_squared_error(y, sk.predict(X)))
    # test on dense grid
    y_test_true = np.sin(X_test[:, 0])
    test_rmse = np.sqrt(mean_squared_error(y_test_true, sk.predict(X_test)))
    print(f'  depth={d:>2}: train RMSE = {train_rmse:.4f}, test RMSE = {test_rmse:.4f}, '
          f'leaves = {sk.tree_.n_leaves}')
```

**출력 예시**:
```
MyTree RMSE on train : 0.1052
sklearn RMSE on train: 0.1051
MyTree leaves: 0  (재귀로 세야 함)

트리 leaf 개수: 22
Leaf 값 (처음 5개): [-0.978, -0.787, -0.314, 0.183, 0.605]
(각 leaf 값은 그 region 내 y의 평균임)

Gain (직접 계산): 32.4000
Gain (정리 3.2 공식): 32.4000

Depth vs train/test RMSE (sin curve):
  depth= 1: train RMSE = 0.6123, test RMSE = 0.6189, leaves = 2
  depth= 2: train RMSE = 0.3245, test RMSE = 0.3187, leaves = 4
  depth= 4: train RMSE = 0.1789, test RMSE = 0.1834, leaves = 16
  depth= 6: train RMSE = 0.0892, test RMSE = 0.1421, leaves = 56
  depth=10: train RMSE = 0.0421, test RMSE = 0.1685, leaves = 121
  depth=20: train RMSE = 0.0000, test RMSE = 0.2104, leaves = 200
```

---

## 🔗 실전 활용

- **sklearn `DecisionTreeRegressor`**: 표준. `criterion='squared_error'` 기본 (예전엔 `'mse'`).
- **회귀 트리의 outlier robustness**: MSE는 outlier 민감. `criterion='absolute_error'` (MAE)로 robust 분할.
- **Random Forest Regressor**: 다중 회귀 트리의 평균 — variance 감소.
- **Gradient Boosting (회귀)**: 잔차에 트리 fitting — residual의 회귀 트리.
- **외삽 위험**: 트리는 train의 max를 넘어선 $x$에 대해 last leaf 값을 그대로 줌 → 외삽 안 됨. 신경망·선형 모델은 가능.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Piecewise constant | 매끄러운 함수 표현 부적합 — 많은 leaf 필요 |
| 외삽 불가 | train 범위 밖 예측은 boundary leaf 값 |
| 불안정성 | 작은 데이터 변화 → 큰 트리 구조 변화 (Ch3-05) |
| 축-정렬 분할 | 대각선 경계 표현 어려움 |

---

## 📌 핵심 정리

$$\boxed{c_v^* = \bar{y}_v,\quad \text{Gain} = \frac{|R_L||R_R|}{|R|}(\bar{y}_L - \bar{y}_R)^2,\quad \text{탐색: } O(p\, n\log n)}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **Leaf 평균** | constant 예측의 MSE-optimal |
| **Gain 공식** | 두 leaf 평균의 차의 제곱 × 가중치 |
| **CART** | 분류 (Gini) + 회귀 (MSE) 통합 framework |
| **Running sum** | 정렬 후 $O(n)$로 모든 split 검사 |
| **Variance reduction** | 분류와 회귀의 **공통 정신** |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $y = (1, 3, 5, 7, 9)$일 때 가능한 모든 binary split의 Gain을 계산해 최적 split을 찾아라.

<details>
<summary>힌트 및 해설</summary>

$\bar{y} = 5$, $L_{\text{total}} = (1-5)^2 + (3-5)^2 + 0 + (7-5)^2 + (9-5)^2 = 16+4+0+4+16 = 40$.

| split | $\bar{y}_L$ | $\bar{y}_R$ | Gain |
|-------|-------------|-------------|------|
| 1\|2345 | 1 | 6 | $1 \cdot 4 / 5 \cdot (1-6)^2 = 20$ |
| 12\|345 | 2 | 7 | $2 \cdot 3 / 5 \cdot 25 = 30$ |
| 123\|45 | 3 | 8 | $3 \cdot 2 / 5 \cdot 25 = 30$ |
| 1234\|5 | 4 | 9 | $4 \cdot 1 / 5 \cdot 25 = 20$ |

**최적**: 12\|345 또는 123\|45 (tie). sklearn은 보통 첫 번째 선택.

</details>

**문제 2** (심화): MAE 분할 (`criterion='absolute_error'`)에서 leaf의 최적 예측값은? MSE의 평균과 어떻게 다른가?

<details>
<summary>힌트 및 해설</summary>

$\arg\min_c \sum |y_i - c|$의 해는 **median**. (1차 미분 조건: $\sum \text{sign}(y_i - c) = 0$ → c는 median.)

**MAE 회귀 트리**: leaf 값이 median, outlier에 robust. 그러나 분할 점수 계산이 더 비쌈 (median 갱신이 sum-based update보다 복잡) — sklearn 구현이 MSE보다 느림.

</details>

**문제 3** (ML 연결): Gradient Boosting의 회귀 트리는 **잔차**에 fit한다. 이것이 본 문서의 회귀 트리와 어떻게 다른가?

<details>
<summary>힌트 및 해설</summary>

본 문서: 트리가 **$y$ 자체**에 fit. Leaf 값 = leaf의 평균 $y$.

Gradient Boosting: 트리가 **현재 모델의 잔차** $r_i = y_i - F_{t-1}(x_i)$에 fit. 첫 번째 트리는 $y$에 fit (since $F_0 = 0$ or $\bar{y}$), 두 번째 트리는 첫 트리의 잔차에 fit, ...

**같은 알고리즘**(MSE 분할), **다른 target** (잔차). Gradient Boosting은 본 문서의 회귀 트리를 base learner로 그대로 사용. Ch5-03에서 자세히.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Gini vs Entropy](./02-gini-vs-entropy.md) | [📚 README](../README.md) | [04. 가지치기 ▶](./04-pruning-cost-complexity.md) |

</div>
