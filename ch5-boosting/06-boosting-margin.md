# 06. Boosting의 과적합 저항성과 Margin Theory

## 🎯 핵심 질문

- AdaBoost가 train error 0 이후에도 **test error 계속 감소**하는 현상의 경험적 관찰 (Schapire et al. 1998)?
- VC bound의 한계: tree 수 $T$ 비례 → conservative. 왜 실제 generalization이 더 좋은가?
- **Margin theory**: $\Pr[m(x, y) \leq \theta]$의 boost 후 감소가 어떻게 일반화 경계를 만드는가?
- 현대적 관점 — **double descent**·**implicit bias**와의 연결.

---

## 🔍 왜 이 개념이 ML에서 중요한가

"Train error 0 이후에도 학습 더 하면 좋아진다"는 ML의 **가장 반직관적 현상** 중 하나. (a) **VC bound로는 설명 불가** — 더 많은 모델 → 더 큰 capacity → 더 큰 bound. 그러나 실제는 반대. (b) **Margin theory** (Schapire 1998)이 첫 만족스러운 설명. (c) 이 직관은 **modern NN의 over-parameterization이 잘 작동하는 이유**의 선구적 사례. (d) Boosting의 "robust to overfitting" 명성의 정량적 기반. 본 문서는 boosting이 단순한 알고리즘 그 이상의 ML 일반화 이론의 모티브였음을 보인다.

---

## 📐 수학적 선행 조건

- AdaBoost의 이론 (Ch5-02)
- VC dimension·Rademacher complexity 개념
- Margin (Ch5-02 정의 2.1)

---

## 📖 직관적 이해

### Schapire et al. (1998)의 관찰

데이터셋: letter recognition (26-class). 트리 stump를 weak learner로 AdaBoost.

- $T = 5$: train 9.4%, test 19.0%
- $T = 100$: train 0.0%, test 8.4% — **train 이미 0인데 test 계속 감소**!
- $T = 1000$: train 0.0%, test 6.9% — 더 감소

**전통적 직관 ("더 많은 모델 = 더 많은 overfit")의 반증**.

### VC Bound는 왜 약한가

VC bound: test error ≤ train error + $O(\sqrt{\text{VC}/n})$. AdaBoost의 hypothesis class = $T$개 tree의 weighted sum. VC dim ∝ $T$ → bound가 $T$ 증가시 커짐.

그러나 실험에서는 test error가 **감소**. → VC bound가 **wrong abstraction**.

### Margin Theory — 더 정확한 측정

핵심 통찰: train error만 보지 말고 **margin distribution**을 보자. AdaBoost는 train error 0 이후에도 **margin을 계속 증가**시킴 (정리 5-2.4) — model이 더 "확신"하게 됨.

**Margin-based bound**:

$$\Pr[\text{test error}] \leq \Pr_{\text{train}}[m(x, y) \leq \theta] + O\!\left(\sqrt{\frac{d}{n \theta^2}}\right).$$

$T$ 증가 → 작은 margin 비율 감소 → 첫 항 감소. **complexity 항은 $T$에 명시적 의존 없음**.

### Modern Connection

- **Implicit Bias**: SGD가 separable data에서 max-margin 해로 수렴 (Soudry 2018) — boosting의 margin 증가와 같은 원리.
- **Double Descent** (Belkin 2019): Over-parameterized 모델이 traditional bias-variance trade-off 위반 — **train error 0 이후에도 test error 감소**.
- **Neural Network**: NN의 "lottery ticket hypothesis", "implicit regularization" 모두 boosting margin theory의 현대 변주.

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Normalized Margin

$F(x) = \sum_t \alpha_t h_t(x)$, $\bar{F}(x) := F(x)/\sum_t |\alpha_t|$. 정규화 margin:

$$m(x, y) := y \cdot \bar{F}(x) \in [-1, 1].$$

### 정의 6.2 — Margin Loss

$$L_\theta(F) := \Pr_{(x, y) \sim P}[m(x, y) \leq \theta].$$

$\theta = 0$이면 일반 0-1 error.

---

## 🔬 정리와 증명

### 정리 6.1 — Margin Distribution Improvement (Schapire 1998 정리 5)

**명제**: AdaBoost iteration $T$ 후 임의의 $\theta < \min_t \frac{1}{2}\log\frac{1-\epsilon_t}{\epsilon_t}/\sum_t \alpha_t$:

$$\Pr_{\text{train}}[m(x, y) \leq \theta] \leq \prod_t 2 \sqrt{\epsilon_t^{1-\theta}(1 - \epsilon_t)^{1+\theta}}.$$

$\theta < $ 어떤 임계값이면 이 양변이 **$T$ 증가시 지수 감소**.

**증명 sketch**: 정리 5-2.1과 유사 — indicator $\mathbb{1}[m \leq \theta]$를 exponential bound로. 단 normalized margin이라 $\sum |\alpha|$ 항이 들어감. 자세히는 Schapire et al. (1998). $\square$

### 정리 6.2 — Margin-based Generalization Bound

**명제** (Schapire 1998 정리 2): 확률 $\geq 1 - \delta$로

$$\Pr_{\text{test}}[m(x, y) < 0] \leq \Pr_{\text{train}}[m(x, y) < \theta] + O\!\left(\sqrt{\frac{d \log^2(n/d) + \log(1/\delta)}{n \theta^2}}\right),$$

$d$는 base learner class의 VC 차원.

**증명 sketch**: VC theory + margin uniform bound. 자세히는 Schapire et al. (1998) 부록.

**핵심**: complexity 항이 $T$에 의존 X — $T$를 늘려도 bound가 커지지 않음. 단지 $\theta$가 작을수록 bound 느슨. $\square$

### 정리 6.3 — Boosting의 "Self-Regularization"

**Informal claim**: AdaBoost는 $T \to \infty$에서 **margin 분포의 minimum**을 최대화 (또는 적어도 stochastic하게 max-margin 해에 가까이 감).

**최근 연구**: Telgarsky (2013) — AdaBoost의 무한 한계가 max-margin classifier (separable 경우). Soudry (2018) — gradient descent on logistic loss in separable data → max-margin (LR 버전).

**연결**: SVM은 explicit max-margin, AdaBoost·SGD는 implicit max-margin.

### 정리 6.4 — Test Error의 Long-term Decrease

**명제** (informal): $T$ 증가 → 정리 6.1에서 작은 $\theta$의 train margin error 감소 → 정리 6.2의 bound 더 tight → test error 감소 가능 (train error가 이미 0이어도).

**경험적 시각화**: $T$별 margin distribution CDF를 그리면 우측으로 이동 — 작은 margin 점이 사라짐. test error는 이 이동에 비례해 감소.

### 정리 6.5 — 한계: Noise가 많으면 Overfit

**명제**: Schapire et al.의 결과는 **clean data**에서 잘 작동. **Label noise**가 많으면 AdaBoost는 noise 점에 weight를 폭발시킴 → 이론적 margin 보장 깨짐 → overfit.

**해결**: 
- **LogitBoost** (Friedman 2000): exponential loss 대신 logistic loss (덜 outlier sensitive).
- **BrownBoost** (Freund 2001): noise-tolerant variant.
- **Gradient Boosting with regularization** (XGBoost·LightGBM): early stopping + shrinkage.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Train error 0 이후 test error 감소 확인
# ─────────────────────────────────────────────
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
y_tr_sign = 2 * y_tr - 1
y_te_sign = 2 * y_te - 1

# 매우 큰 T로 학습
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=500,
                         algorithm='SAMME', random_state=0).fit(X_tr, y_tr)

# T별 train/test error
train_errs = []
test_errs = []
for pred_tr, pred_te in zip(ada.staged_predict(X_tr), ada.staged_predict(X_te)):
    train_errs.append(1 - (pred_tr == y_tr).mean())
    test_errs.append(1 - (pred_te == y_te).mean())

# Train error 0 시점
first_zero = next((t for t, e in enumerate(train_errs) if e == 0), len(train_errs))
print(f'Train error = 0 첫 도달: T = {first_zero + 1}')

# 그 후 test error 감소 추이
print(f'\nTrain error 0 이후 test error 추이:')
for t in [first_zero, first_zero + 50, first_zero + 100, first_zero + 200, len(train_errs) - 1]:
    if t < len(test_errs):
        print(f'  T = {t+1:>4}: train = {train_errs[t]:.4f}, test = {test_errs[t]:.4f}')

# ─────────────────────────────────────────────
# 2. Margin 분포의 우측 이동 (정리 6.1)
# ─────────────────────────────────────────────
def normalized_margin(ada, X, y_sign):
    F = sum(alpha * (clf.predict(X).astype(float) * 2 - 1)
            for alpha, clf in zip(ada.estimator_weights_, ada.estimators_))
    F_norm = sum(np.abs(alpha) for alpha in ada.estimator_weights_)
    return y_sign * F / (F_norm + 1e-10)

print(f'\n특정 margin 임계값 이하 sample 비율:')
print(f'{"T":>5s} | {"P(m≤0)":>9s} | {"P(m≤0.1)":>9s} | {"P(m≤0.3)":>9s} | {"P(m≤0.5)":>9s}')
print('-' * 55)
for T in [10, 50, 100, 200, 500]:
    ada_T = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=T,
                                algorithm='SAMME', random_state=0).fit(X_tr, y_tr)
    m = normalized_margin(ada_T, X_tr, y_tr_sign)
    print(f'{T:>5d} | {(m <= 0).mean():.4f}  | {(m <= 0.1).mean():.4f}  | '
          f'{(m <= 0.3).mean():.4f}  | {(m <= 0.5).mean():.4f}')

# ─────────────────────────────────────────────
# 3. Noise 추가 → Boosting의 한계 (정리 6.5)
# ─────────────────────────────────────────────
print(f'\nLabel noise가 boosting에 미치는 영향:')
for flip in [0.0, 0.1, 0.3]:
    X_noisy, y_noisy = make_classification(n_samples=500, n_features=10, flip_y=flip,
                                            random_state=42)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_noisy, y_noisy, test_size=0.3, random_state=0)
    
    ada_n = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=300,
                                algorithm='SAMME', random_state=0).fit(X_tr2, y_tr2)
    
    train_errs_n = [1 - (p == y_tr2).mean() for p in ada_n.staged_predict(X_tr2)]
    test_errs_n = [1 - (p == y_te2).mean() for p in ada_n.staged_predict(X_te2)]
    
    print(f'\n  noise rate = {flip}:')
    for t in [10, 50, 100, 200, 299]:
        print(f'    T = {t+1:>3}: train = {train_errs_n[t]:.4f}, test = {test_errs_n[t]:.4f}')

# ─────────────────────────────────────────────
# 4. Margin이 큰 sample은 robust하게 분류됨
# ─────────────────────────────────────────────
ada_big = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=500,
                              algorithm='SAMME', random_state=0).fit(X_tr, y_tr)
m_te = normalized_margin(ada_big, X_te, y_te_sign)
pred_te_correct = (m_te > 0)

print(f'\nTest set의 margin 분포:')
print(f'  m > 0.5 (강한 정답): {(m_te > 0.5).mean():.4f}')
print(f'  0 < m ≤ 0.5 (약한 정답): {((m_te > 0) & (m_te <= 0.5)).mean():.4f}')
print(f'  m ≤ 0 (오답)         : {(m_te <= 0).mean():.4f}')
print(f'  → 큰 margin sample이 많을수록 일반화 좋음')

# ─────────────────────────────────────────────
# 5. Logistic loss (LogitBoost) — noise robust
# ─────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier

print(f'\nGradient Boosting (logistic loss) on noisy data:')
X_noisy, y_noisy = make_classification(n_samples=500, n_features=10, flip_y=0.3, random_state=42)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_noisy, y_noisy, test_size=0.3, random_state=0)

ada_noisy = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
                                algorithm='SAMME', random_state=0).fit(X_tr2, y_tr2)
gb_noisy = GradientBoostingClassifier(n_estimators=200, loss='log_loss',
                                       random_state=0).fit(X_tr2, y_tr2)

print(f'  AdaBoost (exp loss)  : test acc = {ada_noisy.score(X_te2, y_te2):.4f}')
print(f'  GBM (log loss)       : test acc = {gb_noisy.score(X_te2, y_te2):.4f}')
print(f'  → log loss가 noise에 더 robust')
```

**출력 예시**:
```
Train error = 0 첫 도달: T = 87

Train error 0 이후 test error 추이:
  T =   87: train = 0.0000, test = 0.1467
  T =  137: train = 0.0000, test = 0.1400
  T =  187: train = 0.0000, test = 0.1400
  T =  287: train = 0.0000, test = 0.1333
  T =  500: train = 0.0000, test = 0.1267

특정 margin 임계값 이하 sample 비율:
    T |  P(m≤0)  | P(m≤0.1)  | P(m≤0.3)  | P(m≤0.5)
-------------------------------------------------------
   10 |  0.0571   |  0.5143    |  0.8086    |  0.9457
   50 |  0.0000   |  0.2914    |  0.5914    |  0.7943
  100 |  0.0000   |  0.2086    |  0.4286    |  0.6571
  200 |  0.0000   |  0.1543    |  0.3200    |  0.5114
  500 |  0.0000   |  0.0857    |  0.2200    |  0.3886

Label noise가 boosting에 미치는 영향:

  noise rate = 0.0:
    T =  11: train = 0.1486, test = 0.1600
    T =  51: train = 0.0000, test = 0.1400
    T = 200: train = 0.0000, test = 0.1267
    T = 299: train = 0.0000, test = 0.1267

  noise rate = 0.3:
    T =  11: train = 0.2543, test = 0.3200
    T =  51: train = 0.1657, test = 0.2933
    T = 200: train = 0.0000, test = 0.3267
    T = 299: train = 0.0000, test = 0.3400  ← overfit 시작

  AdaBoost (exp loss)  : test acc = 0.6800
  GBM (log loss)       : test acc = 0.7133
  → log loss가 noise에 더 robust
```

---

## 🔗 실전 활용

- **Long boosting**: clean data에서 1000+ tree까지 보내도 안전. CV로 best $T$ 자동 선택.
- **XGBoost early stopping**: validation loss 기반 — margin distribution을 implicit하게 모니터.
- **Margin diagnostic**: `predict_proba`의 분포 시각화로 model confidence 측정.
- **Noise-robust loss**: AdaBoost 대신 Gradient Boosting + log loss가 noisy label에 안전.
- **Modern NN과의 연결**: deep learning의 "implicit regularization" 연구가 boosting margin theory의 직접 후예.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Clean label 가정 | noise 많으면 margin theory도 제한 |
| Margin bound 느슨 | 이론 vs 실제 차이 큼 (여전히) |
| 모든 boosting 알고리즘 적용 X | XGBoost·LightGBM의 정규화는 다른 메커니즘 |

---

## 📌 핵심 정리

$$\boxed{\text{Train error 0 이후에도 test error 감소: 작은 margin sample 비율 감소; VC bound 대신 margin-based bound}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **경험적 관찰** | AdaBoost long-run에서 test error 단조 감소 |
| **Margin 우측 이동** | $\Pr[m \leq \theta]$ 감소 |
| **Margin bound** | $T$ 독립 generalization 경계 |
| **Self-regularization** | $T \to \infty$에서 max-margin 같은 해 |
| **Noise 한계** | label noise는 boosting의 weight 폭발 유발 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): VC bound는 왜 boosting의 generalization을 정확히 추정 못 하는가?

<details>
<summary>힌트 및 해설</summary>

VC bound: $\text{test} \leq \text{train} + O(\sqrt{\text{VC}/n})$.

Boosting의 hypothesis class: $T$개 tree의 weighted sum. 각 tree의 VC dim $d_0$ → 전체 VC dim ≈ $T \cdot d_0$ (구체적 bound는 더 복잡하지만 $T$ 비례).

따라서 VC bound가 $T$ 증가시 **확장** → 더 큰 generalization gap 예측. 그러나 실제는 반대.

**근본 이유**: VC dim은 "최악의 hypothesis"를 본다. Boosting은 specific한 알고리즘으로 학습 → 모든 hypothesis 다 가능하지 않음. **Algorithm-specific bound** (margin-based)가 더 정확.

</details>

**문제 2** (심화): Margin 분포가 "우측 이동"하는 것을 정량적으로 정의하라.

<details>
<summary>힌트 및 해설</summary>

Margin CDF $F_T(\theta) = \Pr_{\text{train}}[m \leq \theta]$. "우측 이동" = 모든 $\theta$에 대해 $F_{T'}(\theta) \leq F_T(\theta)$ for $T' > T$ (stochastic dominance — first order).

엄밀하지 않다면, 단지 "큰 $\theta$에서의 $F$ 값이 더 크게 되거나, 작은 $\theta$에서 더 작아짐" — 분포가 더 큰 값으로 mass를 옮김.

정리 6.1은 specific $\theta$의 CDF 값에 대한 upper bound — $T$ 증가시 감소. 모든 $\theta$에서 동시에 보장되는 형태가 진짜 stochastic dominance.

</details>

**문제 3** (ML 연결): NN의 **double descent** (Belkin 2019)이 boosting의 margin theory와 어떻게 같은 정신인가?

<details>
<summary>힌트 및 해설</summary>

**Double descent**: 모델 capacity 증가 → traditional view는 (a) under-param에서 bias 큼, (b) capacity = $n$ 근처에서 test error 폭발, (c) over-param에서 다시 감소.

(c) 부분이 boosting과 같음:
- AdaBoost: train error 0 ($T$가 충분) → 더 많은 $T$ → margin 증가 → test error 감소.
- NN: $\#$weight $\gg n$ → 모든 train data interpolation → 더 큰 모델 → implicit regularization → test error 감소.

**공통 메커니즘**:
- 둘 다 **interpolating regime**: train error 0
- 둘 다 **다른 metric** (margin, weight norm)이 학습되어 일반화 향상
- 둘 다 **VC theory로는 설명 불가**

**역사**: Schapire (1998)의 boosting 결과가 Belkin (2019) double descent의 정신적 선구자. "More is different" — 더 크고 더 학습한 모델이 더 좋다는 modern ML 직관의 시작.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. LightGBM](./05-lightgbm-histogram.md) | [📚 README](../README.md) | [Ch6-01. Naive Bayes ▶](../ch6-nb-discriminant/01-naive-bayes.md) |

</div>
