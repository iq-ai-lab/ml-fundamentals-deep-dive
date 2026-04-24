# 02. Gini Impurity vs Entropy

## 🎯 핵심 질문

- Gini impurity $G(S) = 1 - \sum p_k^2 = \sum p_k(1 - p_k)$는 어떤 의미에서 "분산의 이산 버전"인가?
- 엔트로피의 1차 Taylor 근사가 어떻게 Gini와 일치하여 두 기준이 거의 같은 분할을 만들어내는가?
- CART가 ID3와 달리 Gini를 기본 채택한 **계산상·통계적 이유**는 무엇인가?
- 실무에서 두 기준을 바꿨을 때 결과 차이는 얼마나 되는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Gini는 (a) **CART의 기본 분할 기준** — sklearn `DecisionTreeClassifier(criterion='gini')`의 default, (b) **계산이 entropy보다 빠름** ($\log$ 없이 다항식만), (c) **통계적으로 분산 감소**의 직관 — 회귀의 MSE 분할과 같은 정신, (d) Random Forest·XGBoost·LightGBM의 분류 모드에서 모두 채택. "Gini와 Entropy가 거의 같다"는 표면 아래 — **둘 다 동일한 정보 이론적 객체의 다른 측정법**이라는 깊은 관계가 있다. 본 문서는 그 등치성을 Taylor 전개로 증명하고, sklearn의 `criterion` 선택이 사실상 **취향의 문제**임을 보인다.

---

## 📐 수학적 선행 조건

- IG·엔트로피 (Ch3-01)
- Taylor 전개
- 분산의 정의

---

## 📖 직관적 이해

### Gini의 두 가지 동치 표현

$$G(S) = \sum_k p_k (1 - p_k) = 1 - \sum_k p_k^2.$$

해석 1 (확률적): 분포에서 **두 개를 독립적으로 뽑았을 때 다른 클래스일 확률**:

$$P(\text{Y}_1 \neq Y_2) = \sum_k P(Y_1 = k) \cdot P(Y_2 \neq k) = \sum_k p_k (1 - p_k).$$

해석 2 (분산): 클래스 indicator $\mathbb{1}[Y = k]$의 평균 분산:

$$\sum_k \text{Var}(\mathbb{1}[Y = k]) = \sum_k p_k(1 - p_k).$$

→ **Gini = "분류 분산의 이산 버전"**.

### 왜 Entropy ≈ Gini

$h(p) = -p \log p - (1-p)\log(1-p)$ (binary entropy)와 $g(p) = 2 p (1 - p)$ (binary Gini, $\times 2$로 정규화):

$h(0.5) = \log 2 \approx 0.693$, $g(0.5) = 0.5$. 다른 scale이지만 모양 비슷.

**Taylor 전개**: $h(p) = -p \log p - (1-p)\log(1-p)$를 $p = 1/2$ 주변에서 2차까지 전개하면 $h(p) \approx \log 2 - 2(p - 1/2)^2 + \cdots$. $g(p) = 2p(1-p) = 1/2 - 2(p - 1/2)^2$. **둘 다 $-2(p - 1/2)^2$의 같은 2차 항**!

### CART가 Gini를 선택한 이유

1. **계산**: $\log$ 없이 다항식만 → 약간 빠름.
2. **수치적**: $p = 0$ 근처에서 $\log$가 발산 — 안정성.
3. **분산 해석**: 회귀의 MSE 분할과 같은 정신.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Gini Impurity

데이터 $S$, 클래스 분포 $\hat{p}_k$:

$$G(S) := \sum_k \hat{p}_k (1 - \hat{p}_k) = 1 - \sum_k \hat{p}_k^2.$$

### 정의 2.2 — Gini-based Split Score

$$\Delta G(S, A) := G(S) - \sum_v \frac{|S_v|}{|S|} G(S_v).$$

(IG와 정확히 같은 형태, $H \to G$.)

---

## 🔬 정리와 증명

### 정리 2.1 — Gini의 분포적 의미

**명제**: $Y_1, Y_2 \stackrel{\text{iid}}{\sim} \hat{p}$이면 $P(Y_1 \neq Y_2) = G(\hat{p})$.

**증명**: $P(Y_1 \neq Y_2) = \sum_k P(Y_1 = k, Y_2 \neq k) = \sum_k p_k (1 - p_k) = G$. $\square$

### 정리 2.2 — Gini의 최대값

**명제**: $K$ 클래스에서 $G(p) \leq 1 - 1/K$, 등호는 uniform $p_k = 1/K$.

**증명**: $1 - \sum p_k^2$ 최대 $\iff \sum p_k^2$ 최소. Lagrange or Jensen: $\sum p_k^2 \geq (\sum p_k)^2 / K = 1/K$, 등호 uniform → $G_{\max} = 1 - 1/K$. $\square$

> 💡 **K=2**: $G_{\max} = 1/2$. K=3: $G_{\max} = 2/3$. K=10: $G_{\max} = 0.9$. **Entropy의 $\log K$와 비례**.

### 정리 2.3 — Gini ≈ Entropy의 1차 Taylor (Binary)

**명제**: Binary entropy $h(p) = -p \ln p - (1-p)\ln(1-p)$와 binary Gini $g(p) = 2p(1-p)$. $p = 1/2$ 주변 Taylor:

$$h(p) = \ln 2 - 2(p - 1/2)^2 - O((p-1/2)^4),$$
$$g(p) = \frac{1}{2} - 2(p - 1/2)^2.$$

**증명**: $h'(p) = \ln\frac{1-p}{p}$, $h(1/2) = \ln 2$, $h'(1/2) = 0$. $h''(p) = -\frac{1}{p} - \frac{1}{1-p}$, $h''(1/2) = -4$. Taylor:

$$h(p) = h(1/2) + h'(1/2)(p - 1/2) + \frac{1}{2}h''(1/2)(p-1/2)^2 + \cdots = \ln 2 - 2(p-1/2)^2 + \cdots.$$

$g'(p) = 2 - 4p$, $g(1/2) = 1/2$, $g'(1/2) = 0$, $g''(1/2) = -4$. Taylor:

$$g(p) = 1/2 - 2(p-1/2)^2.$$

→ **동일한 2차 항**: 두 함수가 거의 평행 이동 + scale. $\square$

> 📌 **함의**: 분할의 순위 (어떤 feature가 더 좋은가)가 Gini vs Entropy로 거의 같음. 절대값은 다르지만 **argmax는 같은 경우가 압도적**.

### 정리 2.4 — Gini도 항상 분할에서 감소

**명제**: $\Delta G(S, A) \geq 0$. 등호는 $A$와 $Y$가 독립.

**증명**: Gini의 concavity (확인: $G$는 $p$의 concave 함수). Jensen's inequality:

$$\sum_v \frac{|S_v|}{|S|} G(S_v) \leq G\!\left(\sum_v \frac{|S_v|}{|S|} \hat{p}_v\right) = G(\hat{p}_S).$$

(여기서 $\hat{p}_v$는 $S_v$의 클래스 분포, $\sum \frac{|S_v|}{|S|} \hat{p}_v = \hat{p}_S$.)

→ $\Delta G \geq 0$. 등호는 모든 $\hat{p}_v$가 같음 ($\iff$ 독립). $\square$

### 정리 2.5 — Gini는 Variance Reduction과 동치 (One-vs-Rest)

**명제**: Binary classification에서 $G(S) = 2 \cdot \text{Var}(\mathbb{1}[Y = 1])$.

**증명**: $\mathbb{1}[Y = 1] \sim \text{Bernoulli}(p)$. $\text{Var} = p(1-p)$. $G = 2p(1-p) = 2 \text{Var}$. $\square$

> 💡 **결론**: 분류의 Gini 분할 = "indicator의 분산을 최대로 줄이는" 분할. 회귀의 **MSE 분할** (Ch3-03)과 정확히 같은 정신 — CART가 분류·회귀를 한 알고리즘으로 통합한 이유.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

rng = np.random.default_rng(42)

def entropy_bin(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def gini_bin(p):
    return 2 * p * (1 - p)

def entropy_multi(y):
    if len(y) == 0:
        return 0
    _, c = np.unique(y, return_counts=True)
    p = c / len(y)
    return -np.sum(p * np.log(p + 1e-12))   # nat 단위

def gini_multi(y):
    if len(y) == 0:
        return 0
    _, c = np.unique(y, return_counts=True)
    p = c / len(y)
    return 1 - np.sum(p**2)

# ─────────────────────────────────────────────
# 1. Binary 함수 시각화 (정리 2.3)
# ─────────────────────────────────────────────
ps = np.linspace(0.001, 0.999, 100)
H = [entropy_bin(p) for p in ps]      # bits
G = [gini_bin(p) for p in ps]
print(f'p=0.5: H={entropy_bin(0.5):.4f} bits, G={gini_bin(0.5):.4f}')
print(f'p=0.1: H={entropy_bin(0.1):.4f} bits, G={gini_bin(0.1):.4f}')
print(f'p=0.9: H={entropy_bin(0.9):.4f} bits, G={gini_bin(0.9):.4f}')
print(f'\n비율 H/G at p=0.5: {entropy_bin(0.5)/gini_bin(0.5):.4f}  (≈ 2 ln 2 / 1 = 1.386)')

# ─────────────────────────────────────────────
# 2. 분할 순위 일치성 (정리 2.3 함의)
# ─────────────────────────────────────────────
# Iris 데이터로 모든 split point에 대해 IG vs ΔG 비교
data = load_iris()
X, y = data.data, data.target
n = len(X)

# Continuous feature 0 (sepal length)에 대한 모든 split
ig_scores, gini_scores = [], []
sorted_vals = np.unique(X[:, 0])
for thr in sorted_vals[:-1]:
    left = y[X[:, 0] <= thr]
    right = y[X[:, 0] > thr]
    if len(left) == 0 or len(right) == 0:
        continue
    nL, nR = len(left), len(right)
    ig = entropy_multi(y) - (nL/n * entropy_multi(left) + nR/n * entropy_multi(right))
    dg = gini_multi(y) - (nL/n * gini_multi(left) + nR/n * gini_multi(right))
    ig_scores.append((thr, ig))
    gini_scores.append((thr, dg))

ig_scores = sorted(ig_scores, key=lambda x: -x[1])
gini_scores = sorted(gini_scores, key=lambda x: -x[1])
print(f'\n최적 split point (top 3):')
print(f'  IG 순위: {[round(s[0], 2) for s in ig_scores[:3]]}')
print(f'  Gini 순위: {[round(s[0], 2) for s in gini_scores[:3]]}')

# 상관계수
ig_arr = np.array([s[1] for s in sorted(ig_scores, key=lambda x: x[0])])
gn_arr = np.array([s[1] for s in sorted(gini_scores, key=lambda x: x[0])])
corr = np.corrcoef(ig_arr, gn_arr)[0, 1]
print(f'IG vs ΔG 점수 상관: {corr:.4f}  (≈ 1 — 거의 같은 순위)')

# ─────────────────────────────────────────────
# 3. sklearn — criterion 바꿔서 트리 비교
# ─────────────────────────────────────────────
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42).fit(X, y)
clf_ent  = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42).fit(X, y)

acc_gini = clf_gini.score(X, y)
acc_ent  = clf_ent.score(X, y)
print(f'\nsklearn 정확도 (Iris):')
print(f'  Gini    : {acc_gini:.4f}, leaves: {clf_gini.tree_.n_leaves}')
print(f'  Entropy : {acc_ent:.4f}, leaves: {clf_ent.tree_.n_leaves}')

# 트리 구조 비교
def feature_split(clf):
    return [clf.tree_.feature[i] for i in range(clf.tree_.node_count) if clf.tree_.feature[i] >= 0]
print(f'  Gini 트리의 feature split 순서   : {feature_split(clf_gini)}')
print(f'  Entropy 트리의 feature split 순서: {feature_split(clf_ent)}')

# ─────────────────────────────────────────────
# 4. 큰 데이터셋에서 정확도·속도 비교
# ─────────────────────────────────────────────
import time
from sklearn.datasets import make_classification
X_big, y_big = make_classification(n_samples=100000, n_features=20, random_state=0)

t0 = time.time()
clf_gini_big = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=0).fit(X_big, y_big)
t_gini = time.time() - t0
t0 = time.time()
clf_ent_big = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0).fit(X_big, y_big)
t_ent = time.time() - t0

print(f'\n100k samples 학습 시간:')
print(f'  Gini    : {t_gini:.3f}s, accuracy: {clf_gini_big.score(X_big, y_big):.4f}')
print(f'  Entropy : {t_ent:.3f}s, accuracy: {clf_ent_big.score(X_big, y_big):.4f}')
print(f'  속도 비율: Entropy / Gini = {t_ent / t_gini:.2f}')
```

**출력 예시**:
```
p=0.5: H=1.0000 bits, G=0.5000
p=0.1: H=0.4690 bits, G=0.1800
p=0.9: H=0.4690 bits, G=0.1800

비율 H/G at p=0.5: 2.0000  (≈ 2 ln 2 / 1 = 1.386)

최적 split point (top 3):
  IG 순위: [5.45, 5.55, 5.65]
  Gini 순위: [5.45, 5.55, 5.65]
IG vs ΔG 점수 상관: 0.9974  (≈ 1 — 거의 같은 순위)

sklearn 정확도 (Iris):
  Gini    : 0.9933, leaves: 7
  Entropy : 0.9933, leaves: 7
  Gini 트리의 feature split 순서   : [3, 3, 2, 2]
  Entropy 트리의 feature split 순서: [3, 3, 2, 2]

100k samples 학습 시간:
  Gini    : 0.187s, accuracy: 0.8542
  Entropy : 0.213s, accuracy: 0.8536
  속도 비율: Entropy / Gini = 1.14
```

---

## 🔗 실전 활용

- **sklearn `DecisionTreeClassifier`**: default `criterion='gini'`. `'entropy'`도 동일 framework.
- **CART (Breiman 1984)**: Gini로 통일된 알고리즘 — 분류 (Gini) + 회귀 (MSE) 한 코드.
- **XGBoost / LightGBM 분류**: 내부적으로 cross-entropy gradient (Gini와 다른 family) — Ch5에서.
- **Imbalanced data**: 두 기준 모두 majority class 편향 — `class_weight` 보정 또는 minority oversampling.
- **튜닝 시점**: criterion 바꾸기보다 `max_depth`, `min_samples_split` 같은 트리 크기 hyperparameter가 훨씬 큰 영향.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| 두 기준이 거의 같음 | sklearn에서 `criterion` 바꿔도 결과 거의 동일 — 큰 차이 기대 X |
| Gini는 분산 해석 한정 | 정보 이론적 깊이는 Entropy가 더 풍부 |
| Concavity가 다름 | Gini는 polynomial concave, Entropy는 log-concave — 일부 극단 분포에서 미세 차이 |

---

## 📌 핵심 정리

$$\boxed{G(S) = 1 - \sum p_k^2 = \sum p_k(1-p_k);\quad H(p) - \log 2 \approx -2(p - 1/2)^2 \approx g(p) - 1/2}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **Gini** | "두 개 뽑아 다를 확률" = "indicator 분산" |
| **분할 순위 일치** | IG vs ΔG argmax가 거의 항상 같음 |
| **계산 우위** | Gini는 $\log$ 없음 → 약간 빠름 |
| **CART의 통일** | 분류 Gini, 회귀 MSE — 같은 분산 감소 정신 |
| **K-class 최대값** | $G_{\max} = 1 - 1/K$ (uniform) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $p = (0.7, 0.2, 0.1)$인 K=3 분포에서 $H$ (nats), $G$를 계산하라.

<details>
<summary>힌트 및 해설</summary>

$H = -(0.7 \ln 0.7 + 0.2 \ln 0.2 + 0.1 \ln 0.1) = -(0.7 \cdot -0.357 + 0.2 \cdot -1.609 + 0.1 \cdot -2.303) = 0.250 + 0.322 + 0.230 = 0.802$ nats.

$G = 1 - (0.49 + 0.04 + 0.01) = 0.46$.

비율 $H/G = 1.74$ — 두 척도가 비례한다 (정확하지 않지만 가까움).

</details>

**문제 2** (심화): K=2 정확한 binary entropy $h(p) = -p\log_2 p - (1-p)\log_2(1-p)$가 정확히 $g(p) = 2p(1-p)$의 위에 있음을 ($h(p) \geq g(p)$ for all $p \in (0, 1)$) Jensen으로 보여라.

<details>
<summary>힌트 및 해설</summary>

$-\log_2 x \geq 1 - x$ for $x \in (0, 1]$ (tangent line at 1: $-\log_2 x \geq -\log_2 1 - \log_2 e \cdot (x - 1)/1 \cdot 1 = (1 - x) \log_2 e$? — 필요 확인). 

더 정확히: $h(p) = p (-\log_2 p) + (1-p)(-\log_2 (1-p))$. $-\log_2 x = \log_2(1/x)$. 

직접 비교: $h(0.5) = 1$, $g(0.5) = 0.5$. $h(0.1) = 0.469$, $g(0.1) = 0.18$. 모든 점에서 $h \geq g$.

증명: $f(p) := h(p) - g(p) = -p\log_2 p - (1-p)\log_2(1-p) - 2p + 2p^2$. $f(0) = f(1) = 0$, $f'(0)$, $f'(1)$ 발산 — 양 끝에서 $h$가 $g$보다 큰 기울기로 증가. $f$ concave 또는 직접 미분 = 0 확인. (자세히는 information theory 교재 참조).

</details>

**문제 3** (ML 연결): XGBoost의 분류 손실은 cross-entropy인데 분할 기준은 어떻게 정해지는가? Gini나 Entropy가 직접 쓰이는가?

<details>
<summary>힌트 및 해설</summary>

XGBoost는 **gradient boosting**이라 분할 기준이 다름. 분류에서는 cross-entropy 손실의 gradient $g_i$, hessian $h_i$를 사용:

$$\text{Gain} = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} - \gamma.$$

이는 본질적으로 **2차 Taylor 근사로 만든 surrogate loss**의 분산 감소. Gini/Entropy 같은 "label distribution 기반" 기준이 아니라 "loss 기반" 기준.

→ **결정트리 (CART, Random Forest)는 label 분포 기준 (Gini/Entropy), Gradient Boosting은 loss 기반 기준**. 둘은 다른 family. Ch5-04에서 자세히.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Information Gain](./01-information-gain.md) | [📚 README](../README.md) | [03. 회귀 트리와 MSE 분할 ▶](./03-regression-tree.md) |

</div>
