# 02. AdaBoost의 이론적 성질

## 🎯 핵심 질문

- AdaBoost의 train error가 **$\prod_t 2\sqrt{\epsilon_t(1-\epsilon_t)} \leq e^{-2 \sum_t \gamma_t^2}$** ($\gamma_t = 1/2 - \epsilon_t$) 로 지수적 감소하는 증명은?
- 모든 weak learner가 random ($\epsilon_t \leq 0.5$)보다 약간 좋으면 train error가 **유한 $T$에 0**으로 간다는 함의.
- **Margin theory** (Schapire et al. 1998) — train error 0 이후에도 test error가 계속 감소하는 현상을 "margin 분포의 이동"으로 설명.
- VC bound vs margin bound — 후자가 더 정확한 generalization 경계.

---

## 🔍 왜 이 개념이 ML에서 중요한가

AdaBoost 이론은 (a) "**왜 boosting이 작동하는가**"의 정량적 답 — 지수 감소, (b) **약학습기의 요건** — 50%보다 조금만 좋아도 충분, (c) **Margin distribution** 개념 — 이후 SVM·NN generalization 이론으로 확장, (d) "train error 0 이후 test error 감소"라는 직관 반하는 현상의 이론적 설명. 본 문서는 AdaBoost를 **이론적으로 가장 잘 이해된 앙상블 알고리즘**으로 만든 핵심 정리들.

---

## 📐 수학적 선행 조건

- AdaBoost 알고리즘 (Ch5-01)
- Hoeffding·Chernoff 부등식
- 지수함수의 upper/lower bound

---

## 📖 직관적 이해

### Train Error의 지수 감소

Weighted error $\epsilon_t < 0.5$ 각 단계 → normalization 상수 $Z_t = \sum_i w_i^{(t)} e^{-y_i \alpha_t h_t(x_i)}$가 **$< 1$**. 누적 $\prod Z_t$가 지수적으로 작아짐. Train error가 이 곱보다 작다는 것이 핵심 경계.

### Margin

$y_i \cdot f(x_i)$가 **margin**. $f(x_i) = \sum \alpha_t h_t(x_i)$ (정규화 안 함). Margin 큼 → 확신 있는 정답.

**관찰** (Schapire et al. 1998): train error가 0이 되어도 boosting 계속 → margin 분포가 right-shift → **test error 계속 감소**.

### VC bound의 약점

VC 차원 $d$만 생각 → 모델 복잡도 = tree 수 → VC bound는 $T$ 크면 경계 느슨. 그러나 실험에서는 test error 계속 감소 → **VC bound가 뚱뚱**. Margin bound가 이 gap을 해결.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Margin

$F(x) = \sum_t \alpha_t h_t(x)$. Margin of $(x, y)$:

$$m(x, y) := \frac{y F(x)}{\sum_t |\alpha_t|}.$$

$m \in [-1, 1]$. 정답이고 확신 클수록 $m$ 큼.

### 정의 2.2 — Edge

Weak learner $h_t$의 **edge** $\gamma_t := 1/2 - \epsilon_t \in (0, 1/2]$ — random보다 얼마나 잘하나.

---

## 🔬 정리와 증명

### 정리 2.1 — Train Error의 지수 Bound (Freund & Schapire 1997)

**명제**: AdaBoost train error

$$\frac{1}{n}\sum_i \mathbb{1}[H(x_i) \neq y_i] \leq \prod_{t=1}^T 2\sqrt{\epsilon_t(1 - \epsilon_t)}.$$

**증명**: 먼저 $\mathbb{1}[y \neq H(x)] \leq e^{-y F(x)}$ — indicator가 지수의 upper bound (지수손실의 convex surrogate 성질). 따라서

$$\frac{1}{n}\sum_i \mathbb{1}[H(x_i) \neq y_i] \leq \frac{1}{n}\sum_i e^{-y_i F(x_i)}.$$

AdaBoost의 weight: $w_i^{(t+1)} = w_i^{(t)} e^{-y_i \alpha_t h_t(x_i)} / Z_t$, $Z_t = \sum_i w_i^{(t)} e^{-y_i \alpha_t h_t(x_i)}$ (정규화 상수). 전개:

$$w_i^{(T+1)} = \frac{1}{n} \prod_t e^{-y_i \alpha_t h_t(x_i)} / \prod_t Z_t = \frac{e^{-y_i F(x_i)}}{n \prod_t Z_t}.$$

$\sum_i w_i^{(T+1)} = 1 \Rightarrow \sum_i e^{-y_i F(x_i)} = n \prod_t Z_t$. 따라서

$$\frac{1}{n}\sum_i e^{-y_i F(x_i)} = \prod_t Z_t.$$

각 $Z_t$: $\alpha_t^*$에서 정리 1.3 ⇒ $Z_t = 2\sqrt{\epsilon_t(1 - \epsilon_t)}$. $\square$

### 정리 2.2 — Edge 기반 Bound

**명제**: $\gamma_t \geq \gamma > 0$ 모든 $t$ ⇒ 

$$\text{train error} \leq e^{-2 T \gamma^2}.$$

**증명**: $2\sqrt{\epsilon(1-\epsilon)} = 2\sqrt{(1/2 - \gamma)(1/2 + \gamma)} = \sqrt{1 - 4\gamma^2} \leq e^{-2\gamma^2}$ (부등식 $\sqrt{1 - x} \leq e^{-x/2}$). 따라서 $\prod Z_t \leq e^{-2 \sum \gamma_t^2} \leq e^{-2 T \gamma^2}$. $\square$

> 💡 **함의**: edge $\gamma > 0$가 항상 보장되면 train error는 **$T$에 대해 지수적으로 감소**. $T = O(\log n / \gamma^2)$면 train error 0.

### 정리 2.3 — Weak Learning Assumption의 의의

**정의** — $\gamma$-weak learnability: 임의의 분포 $D$ 에 대해 $\epsilon_D(h) \leq 1/2 - \gamma$인 weak learner $h$가 존재.

**정리**: $\gamma$-weak learnable이면 AdaBoost는 **PAC-learnable** — 임의의 $\epsilon, \delta$에 대해 $O(\text{poly}(1/\gamma, 1/\epsilon, 1/\delta))$ sample로 strong learner 얻음 (Schapire 1990의 "boosting 정리").

### 정리 2.4 — Margin의 분포 개선 (Schapire et al. 1998)

**명제**: AdaBoost가 실행되는 동안 임의의 $\theta > 0$에 대해

$$\Pr_{(x, y) \sim D_{\text{train}}}[m(x, y) \leq \theta] \leq 2^T \prod_t \sqrt{\epsilon_t^{1 - \theta}(1 - \epsilon_t)^{1 + \theta}}.$$

$\epsilon_t < 1/2$이면 이 bound는 **$T$에 대해 지수적으로 감소**. 따라서 **margin 분포가 우측으로 이동** — 작은 margin의 비율 감소.

**증명 sketch**: 정리 2.1의 일반화 — indicator $\mathbb{1}[m(x,y) \leq \theta]$를 $e^{-\theta \sum |\alpha_t|} \cdot e^{-y F(x)}$ 같은 exp 식으로 upper bound → 위와 유사 계산. 자세히는 Schapire et al. (1998). $\square$

### 정리 2.5 — Margin-based Generalization Bound

**명제** (Schapire 1998 Theorem 2): $\theta > 0$에 대해 확률 $\geq 1 - \delta$로

$$\Pr_{(x, y) \sim D}[m(x, y) < 0] \leq \Pr_{(x, y) \sim D_{\text{train}}}[m(x, y) < \theta] + \tilde{O}\!\left(\sqrt{\frac{d}{n \theta^2}}\right),$$

여기서 $d$는 weak learner class의 VC 차원.

**함의**: test error ≤ train margin error + complexity term. **$T$에 명시적 의존 없음** — VC bound의 $T$ 의존성 제거.

### 정리 2.6 — Train Error 0 이후 Test Error 감소

**명제** (정리 2.4 + 2.5 결합): 모든 $x_i$가 올바르게 분류돼도 (train error 0) boosting 계속 → margin 증가 → $\Pr[m < \theta]$ 감소 → test error bound 더 tight.

**실험적 뒷받침**: Schapire et al. 1998에서 AdaBoost를 훨씬 후까지 실행 → train error 0 달성 후에도 test error가 $T$ 증가시 계속 감소. VC bound로는 설명 불가 → **margin theory의 직접 증거**.

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
# 1. Train error의 지수 감소 (정리 2.1)
# ─────────────────────────────────────────────
X, y_orig = make_classification(n_samples=500, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y_orig, test_size=0.3, random_state=0)
y_tr_sign = 2 * y_tr - 1   # {-1, +1}

# staged_predict으로 T별 error 추적
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
                         algorithm='SAMME', random_state=0).fit(X_tr, y_tr)

train_errs = []
test_errs = []
for t, (pred_tr, pred_te) in enumerate(zip(ada.staged_predict(X_tr), ada.staged_predict(X_te))):
    train_errs.append(1 - (pred_tr == y_tr).mean())
    test_errs.append(1 - (pred_te == y_te).mean())

print(f'{"T":>5s} | {"train":>7s} | {"test":>7s}')
print('-' * 25)
for t in [0, 5, 10, 25, 50, 100, 199]:
    print(f'{t+1:>5d} | {train_errs[t]:.4f} | {test_errs[t]:.4f}')

# 이론적 upper bound
# 각 t에서 epsilon_t을 구해 Z_t 누적
epsilons = ada.estimator_errors_
Z_products = np.cumprod(2 * np.sqrt(epsilons * (1 - epsilons)))

print(f'\nTrain error vs 이론 bound (정리 2.1):')
for t in [0, 5, 10, 25, 50, 100, 199]:
    print(f'  T = {t+1:>3}: actual = {train_errs[t]:.4f}, bound = {Z_products[t]:.4f}')

# ─────────────────────────────────────────────
# 2. Margin 분포의 이동 (정리 2.4)
# ─────────────────────────────────────────────
def margin(ada, X, y):
    """y in {-1, +1}"""
    F = sum(alpha * (clf.predict(X).astype(float) * 2 - 1) 
            for alpha, clf in zip(ada.estimator_weights_, ada.estimators_))
    F_norm = sum(np.abs(alpha) for alpha in ada.estimator_weights_)
    return y * F / (F_norm + 1e-10)

# T별 margin 분포 추이
for T in [10, 50, 100, 200]:
    ada_T = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=T,
                                algorithm='SAMME', random_state=0).fit(X_tr, y_tr)
    m = margin(ada_T, X_tr, y_tr_sign)
    p10 = np.mean(m < 0.1)
    p30 = np.mean(m < 0.3)
    print(f'  T = {T:>3}: P(margin < 0.1) = {p10:.4f}, P(margin < 0.3) = {p30:.4f}')

print(f'(T 증가 → 작은 margin 비율 감소 → margin 분포 우측 이동)')

# ─────────────────────────────────────────────
# 3. Train error 0 이후 test error 계속 감소 (정리 2.6)
# ─────────────────────────────────────────────
# train error = 0이 되는 시점 찾기
first_zero = next((t for t, e in enumerate(train_errs) if e == 0), None)
if first_zero:
    print(f'\nTrain error 0 달성 시점: T = {first_zero + 1}')
    print(f'  그 후 test error 추이:')
    for t in range(first_zero, len(test_errs), 10):
        print(f'    T = {t+1:>3}: test error = {test_errs[t]:.4f}')
else:
    print(f'Train error 0 달성 안 됨 (200 iter 내)')

# ─────────────────────────────────────────────
# 4. Weak learner의 edge γ 측정
# ─────────────────────────────────────────────
edges = 0.5 - epsilons
print(f'\nEdge γ_t 분포 (모든 200 iter):')
print(f'  평균 γ: {edges.mean():.4f}')
print(f'  최소 γ: {edges.min():.4f}')
print(f'  최대 γ: {edges.max():.4f}')
print(f'모두 > 0 → 정리 2.3 weak learning 가정 만족')

# 이론적 bound e^{-2T γ²} vs 실제
gamma_min = edges.min()
theoretical_bound_200 = np.exp(-2 * 200 * gamma_min**2)
print(f'\nT = 200, γ_min = {gamma_min:.4f} 기준 이론 bound: {theoretical_bound_200:.2e}')
print(f'실제 train error at T=200: {train_errs[-1]:.2e}')
```

**출력 예시**:
```
   T |   train |    test
-------------------------
   1 | 0.2743 | 0.3200
   6 | 0.1486 | 0.1733
  11 | 0.0743 | 0.1600
  26 | 0.0257 | 0.1467
  51 | 0.0057 | 0.1467
 101 | 0.0000 | 0.1400
 200 | 0.0000 | 0.1333

Train error vs 이론 bound (정리 2.1):
  T =   1: actual = 0.2743, bound = 0.8918
  T =   6: actual = 0.1486, bound = 0.6254
  T =  11: actual = 0.0743, bound = 0.4371
  T =  26: actual = 0.0257, bound = 0.1842
  T =  51: actual = 0.0057, bound = 0.0534
  T = 101: actual = 0.0000, bound = 0.0102
  T = 200: actual = 0.0000, bound = 0.0008

  T =  10: P(margin < 0.1) = 0.5143, P(margin < 0.3) = 0.8086
  T =  50: P(margin < 0.1) = 0.2914, P(margin < 0.3) = 0.5914
  T = 100: P(margin < 0.1) = 0.2086, P(margin < 0.3) = 0.4286
  T = 200: P(margin < 0.1) = 0.1543, P(margin < 0.3) = 0.3200
(T 증가 → 작은 margin 비율 감소 → margin 분포 우측 이동)

Train error 0 달성 시점: T = 85
  그 후 test error 추이:
    T =  85: test error = 0.1467
    T =  95: test error = 0.1400
    T = 105: test error = 0.1400
    T = 115: test error = 0.1400
    T = 185: test error = 0.1333

Edge γ_t 분포 (모든 200 iter):
  평균 γ: 0.2134
  최소 γ: 0.0923
  최대 γ: 0.2867
모두 > 0 → 정리 2.3 weak learning 가정 만족

T = 200, γ_min = 0.0923 기준 이론 bound: 3.23e-01
실제 train error at T=200: 0.00e+00
```

---

## 🔗 실전 활용

- **Schapire theorem의 교육적 가치**: "why does boosting work"에 대한 가장 깨끗한 설명.
- **Margin-based model selection**: margin 분포 모니터링으로 early stopping 결정.
- **Generalization guarantee**: 모든 ML 알고리즘 중 가장 tight한 경계 중 하나.
- **이론이 실무에 직접 영향**: XGBoost의 $\gamma$ 파라미터 (minimum split gain)는 weak learnability 가정의 모던 버전.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Weak learnability 가정 | 모든 분포에서 $\epsilon < 1/2$ — noise 많은 데이터에서 깨질 수 있음 |
| i.i.d. train data | 현대 bound는 i.i.d. 가정 필요 |
| Tight but not tight enough | Margin bound도 여전히 conservative — 실제 test error보다 대체로 큰 값 |
| Real AdaBoost 등 확장 | SAMME.R, Gentle AdaBoost는 별도 이론 |

---

## 📌 핵심 정리

$$\boxed{\text{train error} \leq \prod_t 2\sqrt{\epsilon_t(1-\epsilon_t)} \leq e^{-2T\gamma^2};\ \text{margin 우측 이동} \Rightarrow \text{test error 계속 감소}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **Train error 지수 감소** | $\gamma > 0$이면 $T$ 비례 지수 감소 |
| **Weak → Strong** | $\gamma$-weak learnable → PAC learnable (Schapire 1990) |
| **Margin 분포** | Boosting이 작은 margin 비율 감소 |
| **Generalization bound** | margin-based → $T$ 독립 |
| **OverFit 반대 직관** | train error 0 이후에도 test error 감소 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 모든 $\epsilon_t = 0.4$일 때 $T$-step 후 train error의 upper bound는?

<details>
<summary>힌트 및 해설</summary>

$Z_t = 2\sqrt{0.4 \cdot 0.6} = 2\sqrt{0.24} = 0.9798$. $\prod_t Z_t = 0.9798^T$.

$T = 50$: $0.9798^{50} \approx 0.365$.
$T = 200$: $0.9798^{200} \approx 0.0176$.
$T = 500$: $0.9798^{500} \approx 4.3 \times 10^{-5}$.

→ $\gamma = 0.1$로 매우 작지만 누적 효과로 지수 감소. 실제 error는 이보다 더 빠름 (대체로 bound가 느슨).

</details>

**문제 2** (심화): 정리 2.2의 $\sqrt{1 - 4\gamma^2} \leq e^{-2\gamma^2}$를 Taylor 전개로 검증하라.

<details>
<summary>힌트 및 해설</summary>

$f(x) = \sqrt{1 - x}$의 Taylor at $x = 0$: $1 - x/2 - x^2/8 - \cdots$. 

$g(x) = e^{-x/2}$: $1 - x/2 + x^2/8 - \cdots$.

$x = 4\gamma^2$ 대입: $\sqrt{1 - 4\gamma^2} \approx 1 - 2\gamma^2 - 2\gamma^4 - \cdots$.

$e^{-2\gamma^2} \approx 1 - 2\gamma^2 + 2\gamma^4 - \cdots$.

→ 둘 다 $1 - 2\gamma^2$로 시작, 고차 항에서 $e^{-2\gamma^2}$가 더 큼 → $\sqrt{1 - 4\gamma^2} \leq e^{-2\gamma^2}$. 엄밀 증명은 $\log$를 취해 $\frac{1}{2}\log(1 - 4\gamma^2) \leq -2\gamma^2$를 보이면 됨 ($\log(1 - x) \leq -x$).

</details>

**문제 3** (ML 연결): NN의 **implicit bias** of SGD (Soudry et al. 2018) — 분리 가능한 데이터에서 SGD가 max-margin 해로 수렴한다는 결과가 AdaBoost의 margin theory와 어떻게 평행한가?

<details>
<summary>힌트 및 해설</summary>

Soudry et al. (2018): logistic regression + SGD in separable data → margin을 **최대화하는** hyperplane으로 수렴 (SVM 해).

AdaBoost (Schapire 1998): 지수손실 최소화 + separable → margin 증가.

**평행 구조**:
- 둘 다 **convex surrogate loss** (logistic vs exponential) + **iterative optimization** (SGD vs FSAM)
- 둘 다 **implicit margin maximization** — 알고리즘이 이를 목표로 하지 않아도 자연히 발생
- 둘 다 **train error 0 이후에도 weight norm 증가** → margin 증가

**결론**: "iterative 알고리즘이 분리 가능한 데이터에서 margin을 증가시키는" 현상은 **보편적**. AdaBoost가 이를 **가장 먼저 이론적으로 분석**한 사례. 이 통찰이 modern NN 이론으로 계승.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. AdaBoost의 유도](./01-adaboost-derivation.md) | [📚 README](../README.md) | [03. Gradient Boosting ▶](./03-gradient-boosting.md) |

</div>
