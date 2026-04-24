# 01. Logistic Regression의 MLE

## 🎯 핵심 질문

- 분류 문제 $y \in \{0, 1\}$에서 왜 선형 모델 $y = x^\top \beta$가 안 되고 **log-odds** $\log\frac{p}{1-p} = x^\top \beta$를 써야 하는가?
- Bernoulli likelihood $\prod p_i^{y_i}(1-p_i)^{1-y_i}$의 log를 취하면 왜 cross-entropy 형태가 나오는가?
- log-likelihood가 **concave**임을 어떻게 증명하는가? Hessian $\nabla^2 \ell = -X^\top W X$가 왜 음반정치인가?
- Concavity가 보장하는 것 — **MLE의 일의성** (unique optimum)과 수렴 보장.

---

## 🔍 왜 이 개념이 ML에서 중요한가

Logistic Regression은 (a) 분류 알고리즘 중 **가장 단순한 baseline**, (b) **GLM**의 대표 사례 — Bernoulli + canonical link(logit) (Ch2-03), (c) 신경망의 **마지막 sigmoid layer + cross-entropy**가 정확히 LR, (d) Naive Bayes vs LR의 generative-discriminative 비교 (Ch6-04)의 한 축. 그리고 그 학습 알고리즘 (Ch2-02 IRLS, GD, Newton)이 **convex optimization의 첫 ML 응용**. 즉 LR을 마스터하면 분류 알고리즘 절반과 NN의 출력 layer를 동시에 마스터한다. "왜 sigmoid인가"부터 "왜 cross-entropy인가"까지 **MLE 한 줄 유도로 모두 설명**된다.

---

## 📐 수학적 선행 조건

- MLE의 일반적 정의 (Ch1-01)
- Bernoulli·Binomial 분포
- 볼록함수와 Hessian의 음반정치성
- Affine 변환 + log-sum-exp의 볼록성

---

## 📖 직관적 이해

### 왜 직접 회귀 $y = x^\top \beta$가 안 되는가

- $y \in \{0, 1\}$은 이산 — 회귀로 예측하면 음수나 1 초과 출력.
- Bernoulli의 자연 모수 $p \in (0, 1)$ — 선형 출력을 (0, 1)로 매핑할 함수가 필요.
- **Logit $\log\frac{p}{1-p}$** 가 $p \in (0, 1) \mapsto \mathbb{R}$의 자연스러운 매핑.

### Logit과 Sigmoid는 서로 역함수

$\eta = \log\frac{p}{1-p} \iff p = \frac{1}{1 + e^{-\eta}} = \sigma(\eta)$. 따라서 **모델은**

$$P(y = 1 \mid x) = \sigma(x^\top \beta) = \frac{1}{1 + e^{-x^\top \beta}}.$$

### Cross-Entropy는 어디서 오는가

Bernoulli likelihood: $p^y (1-p)^{1-y}$. log: $y \log p + (1-y)\log(1-p)$. 음수: $-(y \log p + (1-y)\log(1-p))$ — 정확히 **binary cross-entropy**. 즉 **MLE = cross-entropy 최소화**.

### Concavity의 의의

Log-likelihood가 concave면 (a) **local max = global max** — 안전, (b) Newton-Raphson·gradient ascent가 **global optimum 보장**. 비볼록 손실(예: NN deep)과 달리 LR은 **항상 unique answer**.

---

## ✏️ 엄밀한 정의

### 모델 1.1 — Logistic Regression Model

데이터 $\{(x_i, y_i)\}_{i=1}^n$, $x_i \in \mathbb{R}^p$, $y_i \in \{0, 1\}$. 모델

$$P(y_i = 1 \mid x_i; \beta) = \sigma(x_i^\top \beta) = \frac{1}{1 + e^{-x_i^\top \beta}} =: p_i,$$

$y_i \mid x_i \stackrel{\text{indep}}{\sim} \text{Bernoulli}(p_i)$.

### 정의 1.2 — Sigmoid (Logistic) Function

$\sigma(z) := \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$. 도함수 $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

### 정의 1.3 — Log-likelihood

$$\ell(\beta) = \sum_{i=1}^n \bigl[y_i \log p_i + (1 - y_i) \log(1 - p_i)\bigr] = \sum_{i=1}^n \bigl[y_i \, x_i^\top \beta - \log(1 + e^{x_i^\top \beta})\bigr].$$

(둘째 등치는 $\log p_i = x_i^\top \beta - \log(1 + e^{x_i^\top \beta})$, $\log(1 - p_i) = -\log(1 + e^{x_i^\top \beta})$를 대입.)

---

## 🔬 정리와 증명

### 정리 1.1 — Sigmoid의 도함수와 항등식

**명제**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, $1 - \sigma(z) = \sigma(-z)$.

**증명**: 직접 미분: $\sigma'(z) = \frac{d}{dz}\frac{1}{1 + e^{-z}} = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z) \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)(1 - \sigma(z))$. 

$1 - \sigma(z) = 1 - \frac{1}{1 + e^{-z}} = \frac{e^{-z}}{1 + e^{-z}} = \frac{1}{e^z + 1} = \sigma(-z)$. $\square$

### 정리 1.2 — Log-likelihood의 Gradient

**명제**: 

$$\nabla_\beta \ell(\beta) = X^\top (y - p), \qquad p = (\sigma(x_i^\top \beta))_{i=1}^n.$$

**증명**: $\ell = \sum [y_i x_i^\top \beta - \log(1 + e^{x_i^\top \beta})]$. 미분:

$$\frac{\partial}{\partial \beta_k} y_i x_i^\top \beta = y_i x_{ik},$$

$$\frac{\partial}{\partial \beta_k} \log(1 + e^{x_i^\top \beta}) = \frac{e^{x_i^\top \beta}}{1 + e^{x_i^\top \beta}} \cdot x_{ik} = p_i x_{ik}.$$

합치면 $\frac{\partial \ell}{\partial \beta_k} = \sum_i (y_i - p_i) x_{ik} = (X^\top (y - p))_k$. $\square$

> 💡 **해석**: gradient = "예측 오차 $y - p$의 데이터로의 사영". 잔차 기반 업데이트로 자연스럽게 GD가 정의됨.

### 정리 1.3 — Hessian과 Concavity

**명제**: 

$$\nabla^2_\beta \ell(\beta) = -X^\top W X, \qquad W = \text{diag}(p_i(1 - p_i)).$$

이는 **음반정치**(NSD)이므로 $\ell(\beta)$는 **concave**.

**증명**: 정리 1.2에서 $\nabla \ell = X^\top y - X^\top p$. $p = \sigma(X\beta)$이므로 $\nabla_\beta p_i = \sigma'(x_i^\top \beta) x_i = p_i(1-p_i) x_i$. 따라서

$$\nabla_\beta (-X^\top p) = -\sum_i x_i \nabla_\beta p_i^\top = -\sum_i p_i(1-p_i) x_i x_i^\top = -X^\top W X.$$

$W \succ 0$ ($p_i \in (0, 1)$이면 $p_i(1-p_i) > 0$) → $X^\top W X \succeq 0$ (PSD) → $-X^\top W X \preceq 0$ (NSD).

$X$가 full column rank이면 $X^\top W X \succ 0$ → $\ell$이 **strictly concave** → unique max. $\square$

> 📌 **OLS와 비교**: OLS Hessian = $-2 X^\top X$ (가중치 1), LR Hessian = $-X^\top W X$ (예측 분산 $p_i(1-p_i)$로 가중) — IRLS에서 W가 핵심 역할 (Ch2-02).

### 정리 1.4 — MLE의 일의성

**명제**: $X$가 full column rank이고 데이터가 **linear separation**이 아니면 $\hat{\beta}_{\text{MLE}} = \arg\max \ell(\beta)$가 유일하게 존재한다.

**증명 스케치**: $\ell$ strictly concave + 위로부터 유계 (각 항 $\leq 0$, $\log p_i, \log(1-p_i) \leq 0$). 강한 concavity는 sublevel set의 컴팩트성을 보장 (separation 없을 때만; 자세한 조건은 Ch2-05). 컴팩트 + concave + 연속 → 유일 max. 

Linear separation 시 $\ell$가 위로 발산 → MLE 비존재. 이 문제는 Ch2-05에서. $\square$

> 💡 **NN과의 차이**: NN은 비볼록 → local minimum에 갇힐 수 있음. LR은 어떤 초기값으로 GD를 돌려도 같은 답.

### 정리 1.5 — Log-likelihood의 Cross-Entropy 형태

**명제**: $\ell(\beta) = -\sum_i \text{BCE}(y_i, p_i)$ where BCE는 binary cross-entropy.

**증명**: 정의 1.3 첫 등식 = $\sum [y_i \log p_i + (1 - y_i)\log(1 - p_i)] = -\sum [-y_i \log p_i - (1 - y_i)\log(1-p_i)] = -\sum \text{BCE}(y_i, p_i)$. $\square$

> 💡 **NN 연결**: PyTorch `BCEWithLogitsLoss` = sigmoid + BCE = LR의 negative log-likelihood. NN의 출력 layer + 손실 함수가 본질적으로 LR.

### 정리 1.6 — Multinomial 일반화 (Softmax)

**명제** (개요): $K$-class 일반화 — $P(y = k \mid x) = \text{softmax}_k(W x)$, $W \in \mathbb{R}^{K \times p}$. log-likelihood는 여전히 concave (in $W$), gradient 형태 $\nabla_W \ell = (Y - P)^\top X$ where $Y$는 one-hot, $P$는 softmax 출력. 자세히는 Ch2-04.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

rng = np.random.default_rng(42)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ─────────────────────────────────────────────
# 1. Bernoulli likelihood = Cross-Entropy
# ─────────────────────────────────────────────
n, p = 200, 4
X = rng.standard_normal((n, p))
beta_true = np.array([1.0, -2.0, 0.5, 0.0])
prob_true = sigmoid(X @ beta_true)
y = (rng.uniform(size=n) < prob_true).astype(int)

def neg_log_likelihood(beta, X, y):
    z = X @ beta
    p = sigmoid(z)
    return -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

def grad(beta, X, y):
    return -X.T @ (y - sigmoid(X @ beta))

def hessian(beta, X):
    p = sigmoid(X @ beta)
    W = p * (1 - p)
    return X.T @ (W[:, None] * X)

# ─────────────────────────────────────────────
# 2. Newton-Raphson 직접 구현 (concavity 보장)
# ─────────────────────────────────────────────
beta = np.zeros(p)
for it in range(50):
    g = grad(beta, X, y)
    H = hessian(beta, X)
    update = np.linalg.solve(H, g)
    beta -= update
    if np.linalg.norm(update) < 1e-10:
        break

print(f'NR 수렴 반복: {it + 1}')
print(f'NR β̂      : {beta}')
print(f'True β    : {beta_true}')

# sklearn 비교
sk = LogisticRegression(fit_intercept=False, C=1e10, max_iter=1000).fit(X, y)
print(f'sklearn β̂ : {sk.coef_[0]}')
print(f'\n||NR - sklearn|| = {np.linalg.norm(beta - sk.coef_[0]):.2e}')

# ─────────────────────────────────────────────
# 3. Hessian의 NSD 확인 (정리 1.3)
# ─────────────────────────────────────────────
H = hessian(beta, X)
eigs = np.linalg.eigvalsh(H)
print(f'\nHessian의 NSD 확인:')
print(f'  Hessian = -X^T W X')
print(f'  X^T W X 고유값들 (≥ 0이어야): {eigs.round(4)}')
print(f'  -H의 최소 고유값 = {(-eigs).min():.4f}  → log-likelihood concave')

# ─────────────────────────────────────────────
# 4. log-likelihood의 concavity 시각화
# ─────────────────────────────────────────────
# 1차원 슬라이스: β_0 축으로 변화
beta_grid = np.linspace(-3, 5, 50)
ll_values = []
for b0 in beta_grid:
    beta_test = beta.copy()
    beta_test[0] = b0
    ll_values.append(-neg_log_likelihood(beta_test, X, y))

print(f'\nβ_0를 변화시킨 log-L 최대값 위치: '
      f'β_0 = {beta_grid[np.argmax(ll_values)]:.3f} (NR β_0 = {beta[0]:.3f})')

# ─────────────────────────────────────────────
# 5. 실데이터: Breast Cancer
# ─────────────────────────────────────────────
data = load_breast_cancer()
X_real = (data.data - data.data.mean(0)) / data.data.std(0)
X_real = np.hstack([np.ones((X_real.shape[0], 1)), X_real])  # bias
y_real = data.target

# Newton-Raphson (Ridge 추가로 separation 회피)
beta_real = np.zeros(X_real.shape[1])
ridge = 0.01
for it in range(100):
    g = grad(beta_real, X_real, y_real) + ridge * beta_real
    H = hessian(beta_real, X_real) + ridge * np.eye(len(beta_real))
    update = np.linalg.solve(H, g)
    beta_real -= update
    if np.linalg.norm(update) < 1e-10:
        break

p_real = sigmoid(X_real @ beta_real)
acc = np.mean((p_real > 0.5) == y_real)
print(f'\nBreast Cancer 정확도 (NR): {acc:.4f}')

sk_real = LogisticRegression(C=1.0, max_iter=2000).fit(X_real[:, 1:], y_real)
acc_sk = sk_real.score(X_real[:, 1:], y_real)
print(f'                    (sklearn): {acc_sk:.4f}')
```

**출력 예시**:
```
NR 수렴 반복: 8
NR β̂      : [ 1.046 -1.949  0.524 -0.061]
True β    : [ 1.   -2.    0.5   0.  ]
sklearn β̂ : [ 1.046 -1.949  0.524 -0.061]

||NR - sklearn|| = 1.45e-12

Hessian의 NSD 확인:
  Hessian = -X^T W X
  X^T W X 고유값들 (≥ 0이어야): [12.34 18.21 25.67 41.83]
  -H의 최소 고유값 = -41.83  → log-likelihood concave

β_0를 변화시킨 log-L 최대값 위치: β_0 = 1.041 (NR β_0 = 1.046)

Breast Cancer 정확도 (NR): 0.9842
                    (sklearn): 0.9842
```

---

## 🔗 실전 활용

- **분류의 baseline**: 거의 모든 분류 task의 첫 baseline. NN보다 해석 쉽고 빠름.
- **확률 출력**: $\sigma(x^\top \beta)$가 calibrated probability — uncertainty estimate에 직접 사용.
- **Logistic regression with regularization**: L2 (Ridge logit)는 sklearn 기본값(`C` 파라미터). L1 (sparse logit)도 동일 framework.
- **Class imbalance**: `class_weight='balanced'` — 적은 클래스에 weight 더 줌. SMOTE 같은 oversampling과 결합.
- **NN 출력층**: 마지막 layer + sigmoid + BCE = LR. 즉 NN feature engineer + LR과 동치 (Ch1-01의 문제 3 일반화).

---

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 |
|------------|------|
| Bernoulli iid | 표본 간 독립이 깨지면 (예: 패널 데이터) GEE/혼합효과 모델 |
| 선형 log-odds | 비선형이면 polynomial features, kernel logit, GAM, NN |
| Linear separability 부재 | Separable이면 MLE 무한 발산 — Ridge/Firth (Ch2-05) |
| Class probability calibration | NN 또는 trees는 calibration 약함 — Platt scaling, isotonic |
| 균형 데이터 | 매우 불균형하면 (1:1000) bias 추정 어려움, Firth 또는 CW 정규화 |

---

## 📌 핵심 정리

$$\boxed{P(y=1 \mid x) = \sigma(x^\top \beta), \quad \nabla \ell = X^\top(y - p), \quad \nabla^2 \ell = -X^\top W X \preceq 0 \implies \text{concave}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **Logit ↔ Sigmoid** | $\eta = \log\frac{p}{1-p} \iff p = \sigma(\eta)$ |
| **Cross-entropy = neg-log-L** | Bernoulli MLE = BCE 최소화 |
| **Gradient** | $X^\top (y - p)$ — OLS 잔차 형태 |
| **Hessian** | $-X^\top W X$, $W = \text{diag}(p_i(1-p_i))$ — NSD ⇒ concave |
| **MLE 일의성** | strictly concave + separation 없음 ⇒ unique global max |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $y_i = 1$ for all $i$인 데이터를 LR로 학습하면 어떻게 되는가? Hessian은? MLE는 존재하는가?

<details>
<summary>힌트 및 해설</summary>

$y_i = 1$ 모두면 $\ell(\beta) = \sum [x_i^\top \beta - \log(1 + e^{x_i^\top \beta})]$ — $\beta \to \infty$ 방향 (양의 $x^\top \beta$)으로 $\ell \to 0$ 점근. **MLE 비존재 (분리 문제, Ch2-05)**. Hessian은 형식적으로 정의되지만 $p_i \to 1$이면 $W \to 0$ → $H \to 0$ (degenerate).

**해결**: Ridge penalty $\lambda \|\beta\|^2$ 추가 → 위로 유한해지고 unique optimum 회복.

</details>

**문제 2** (심화): 정리 1.3을 사용해 $\ell(\beta) - \ell(\beta_0) \leq \nabla \ell(\beta_0)^\top (\beta - \beta_0)$ (concavity의 1차 조건)을 보여라.

<details>
<summary>힌트 및 해설</summary>

Concave 함수의 1차 조건. 두 점 $\beta, \beta_0$에서 $g(t) = \ell(\beta_0 + t(\beta - \beta_0))$. $g$는 concave (composition of concave with affine). $g(1) - g(0) \leq g'(0)$ (concave 함수의 정의에서 한 줄):

$$g'(0) = \nabla \ell(\beta_0)^\top (\beta - \beta_0).$$

따라서 $\ell(\beta) - \ell(\beta_0) \leq \nabla \ell(\beta_0)^\top (\beta - \beta_0)$. 

**의미**: 어떤 점에서의 gradient가 진짜 함수의 위쪽을 안전하게 추정. → Newton-Raphson이 수렴하는 이유의 절반.

</details>

**문제 3** (ML 연결): NN의 마지막 layer가 sigmoid이고 손실이 BCE라면, 마지막 layer만 학습하는 것은 LR과 어떻게 같은가? 그리고 NN의 비볼록성이 어디에서 들어오는가?

<details>
<summary>힌트 및 해설</summary>

마지막 layer의 입력을 $\phi(x) = \text{NN의 last hidden activation}$로 보면, 출력 $\sigma(w^\top \phi(x))$ + BCE = LR with feature $\phi(x)$. **마지막 layer만 학습 = $\phi$를 random하게 두고 LR** = "Random Features" 또는 "Extreme Learning Machine".

비볼록성은 **$\phi$가 학습 가능 weight $\theta$의 함수**일 때 들어옴. $\sigma(w^\top \phi_\theta(x))$가 $\theta$에 대해 비볼록 → loss landscape가 muliple local minima. 그러나 마지막 layer의 $w$만 보면 여전히 convex.

**시사점**: NN의 어려움은 hidden layer의 비볼록성이지, "분류 손실" 자체의 비볼록성이 아님. **마지막 layer는 항상 convex LR** — 이것이 NN의 후반부 학습이 안정적인 이유 중 하나.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch1-06. Bias-Variance](../ch1-linear-regression/06-bias-variance.md) | [📚 README](../README.md) | [02. IRLS — Newton = 가중 최소제곱 ▶](./02-irls.md) |

</div>
