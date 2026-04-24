# 03. Exponential Family와 Canonical Link

## 🎯 핵심 질문

- **Exponential Family** $p(y; \theta) = h(y) \exp(\theta^\top T(y) - A(\theta))$는 어떻게 Gaussian, Bernoulli, Poisson, Gamma를 **한 틀**에 묶는가?
- GLM의 3구성(분포·선형예측자 $\eta$·link function $g$)을 왜 그렇게 나누는가?
- **Canonical link** ($g = (A')^{-1}$)를 쓰면 왜 **Fisher scoring = IRLS**가 되고 계산이 극적으로 단순해지는가?
- Logit·Probit·Log-log는 어떻게 비교되고, Poisson 회귀·Gamma 회귀는 같은 틀에서 어떻게 나오는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

GLM은 "MLE를 하는 선형 모델"의 **최대 일반화** — Gaussian(OLS)·Bernoulli(LR)·Poisson(count data)·Gamma(양수 continuous)·Multinomial(multi-class)을 **한 알고리즘(IRLS)**과 **한 이론적 틀(Exponential Family)**로 동시에 다룬다. 본 문서의 핵심 — **canonical link**를 쓰면 (a) score equation이 잔차 형태 $X^\top (y - \mu) = 0$, (b) Hessian이 $-X^\top W X$ 형태, (c) IRLS 한 iteration이 매우 단순 — 라는 세 가지 좋은 성질이 자동으로 나온다. 즉 "왜 LR은 sigmoid를, Poisson은 exp를, Gaussian은 identity를 쓰는가"에 대한 **통일 답**이 여기 있다.

---

## 📐 수학적 선행 조건

- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): Fisher 정보, Cramér-Rao 하한, sufficient statistic
- LR의 MLE와 IRLS (Ch2-01, Ch2-02)
- 지수함수의 Legendre 변환 (선택)

---

## 📖 직관적 이해

### "한 분포족"의 정의

Exponential Family는 밀도가 **지수함수 안의 선형 형태**:

$$p(y; \theta) = h(y) \exp\bigl(\theta^\top T(y) - A(\theta)\bigr).$$

- $\theta$: **natural (canonical) parameter**
- $T(y)$: **sufficient statistic**
- $A(\theta) = \log \int h(y) e^{\theta^\top T(y)} dy$: **log-partition function** (normalization constant의 log)
- $h(y)$: base measure (often 1)

### GLM의 3구성

1. **랜덤 부분 (분포)**: $y \mid x$가 지수족의 어떤 분포.
2. **체계 부분 (선형 예측자)**: $\eta = x^\top \beta$ — 선형.
3. **Link function** $g$: $\mu := \mathbb{E}[y \mid x] = g^{-1}(\eta)$. 즉 $g(\mu) = \eta$.

### Canonical Link의 마법

분포의 **natural parameter** $\theta$와 **mean** $\mu$ 사이에 자연스러운 함수 $\theta = \psi(\mu)$가 있다 ($\psi = (A')^{-1}$). **Canonical link**는 바로 $g = \psi$, 즉 $g(\mu) = \theta$. 이 선택은 **$\eta = \theta$**이 되게 하여 **IRLS의 Hessian이 $-X^\top W X$**로 단순화된다 (정리 3.3).

| 분포 | $A(\theta)$ | $\mu = A'(\theta)$ | Canonical link $g(\mu)$ |
|------|-------------|--------------------|-----------------------|
| Normal ($\sigma^2$ 알려짐) | $\theta^2 / 2$ | $\theta$ | identity |
| Bernoulli | $\log(1 + e^\theta)$ | $\sigma(\theta)$ | **logit** $\log \frac{\mu}{1-\mu}$ |
| Poisson | $e^\theta$ | $e^\theta$ | **log** |
| Gamma (shape $k$) | $-\log(-\theta)$ | $-1/\theta$ | **inverse** $1/\mu$ |
| Binomial ($n$ 알려짐) | $n \log(1 + e^\theta)$ | $n \sigma(\theta)$ | logit |

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Exponential Family

$Y$의 밀도가 공통된 형태

$$p(y; \theta) = h(y) \exp\bigl(\theta^\top T(y) - A(\theta)\bigr)$$

인 분포족을 **(one-parameter linear) exponential family**라 한다. $\theta$를 **natural parameter**, $A$를 **log-partition function**이라 한다.

### 정의 3.2 — Generalized Linear Model

GLM은 다음 3요소로 구성된다:

1. **랜덤 부분**: $y \mid x$가 지수족 분포, natural parameter $\theta(x)$, mean $\mu(x) = \mathbb{E}[y \mid x]$.
2. **체계 부분**: $\eta(x) = x^\top \beta$.
3. **Link function**: $g$ (smooth, monotone) s.t. $g(\mu) = \eta$, 즉 $\mu = g^{-1}(\eta)$.

### 정의 3.3 — Canonical Link

$\theta = \psi(\mu)$ (분포족에서 자연스럽게 결정)일 때 $g = \psi$, 즉 $\eta = \theta$로 놓으면 $g$를 **canonical link**라 한다. 

$A'(\theta) = \mu$ → $\psi = (A')^{-1}$. 따라서 canonical link는 $g = (A')^{-1}$.

---

## 🔬 정리와 증명

### 정리 3.1 — Exponential Family의 평균과 분산

**명제**: $y \sim p(\cdot; \theta)$이면

$$\mathbb{E}[T(y)] = A'(\theta), \qquad \text{Var}(T(y)) = A''(\theta).$$

**증명**: $\int p(y; \theta) dy = 1$을 $\theta$로 미분:

$$0 = \int \bigl(T(y) - A'(\theta)\bigr) p(y; \theta) dy = \mathbb{E}[T(y)] - A'(\theta).$$

2차 미분:

$$0 = \int \bigl((T(y) - A'(\theta))^2 - A''(\theta)\bigr) p(y; \theta) dy.$$

→ $\text{Var}(T) = A''(\theta)$. $\square$

> 💡 **예**: Bernoulli에서 $A(\theta) = \log(1 + e^\theta)$ → $A'(\theta) = \sigma(\theta) = p$, $A''(\theta) = p(1-p)$. 익숙한 식.

### 정리 3.2 — Canonical Link 하에서 Score Equation

**명제**: Canonical link($\eta = \theta$)를 사용하는 GLM에서

$$\nabla_\beta \ell(\beta) = X^\top (y - \mu), \qquad \mu = g^{-1}(X\beta) = A'(X\beta).$$

**증명**: log-likelihood $\ell = \sum_i [\theta_i T(y_i) - A(\theta_i)] + \text{const}$ ($T(y) = y$이라 가정 — 대부분의 일반적 경우). Canonical link면 $\theta_i = \eta_i = x_i^\top \beta$. 따라서

$$\ell = \sum_i [y_i x_i^\top \beta - A(x_i^\top \beta)] + \text{const}.$$

미분: $\nabla \ell = \sum_i [y_i x_i - A'(x_i^\top \beta) x_i] = X^\top (y - \mu)$. $\square$

> 📌 **왜 중요한가**: score의 형태가 **"잔차 $y - \mu$의 $X$로의 사영 = 0"** — OLS의 normal equation $X^\top(y - X\beta) = 0$과 **같은 구조**. 모든 GLM에서.

### 정리 3.3 — Canonical Link 하에서 Hessian

**명제**: Canonical link GLM의 Hessian:

$$\nabla^2_\beta \ell(\beta) = -X^\top W X, \qquad W = \text{diag}(A''(\eta_i)) = \text{diag}(\text{Var}(y_i)).$$

**증명**: 정리 3.2의 gradient를 다시 미분. $\partial \mu_i / \partial \beta = A''(\eta_i) x_i$. 따라서

$$\nabla^2 \ell = -\sum_i A''(\eta_i) x_i x_i^\top = -X^\top W X. \quad \square$$

> 💡 **정리 1.3과 정확히 같음**: LR에서 $W = \text{diag}(p_i(1-p_i))$. Poisson에서 $W = \text{diag}(\mu_i)$. Normal에서 $W = I$. **분포만 바뀌고 공식은 그대로**.

### 정리 3.4 — Fisher Scoring = IRLS (Canonical Link)

**명제**: Canonical link GLM에서 Fisher scoring (Hessian을 Fisher information으로 대체한 Newton)는 Newton-Raphson과 **정확히 같고**, IRLS와 동치.

**증명**: Fisher information $I(\beta) = \mathbb{E}[-\nabla^2 \ell] = X^\top W X$ (정리 3.3에서 Hessian이 이미 deterministic — $y$에 의존 안함; canonical 경우 특수). 따라서 Fisher scoring step $\beta \leftarrow \beta + I^{-1} \nabla \ell$ = Newton step. Ch2-02의 IRLS 유도 그대로 적용. $\square$

> 📌 **Canonical link의 큰 장점**: 비-canonical link에서는 Fisher information과 Hessian이 달라 Fisher scoring이 Newton과 **다름**. Canonical이면 둘이 일치 → 알고리즘 단순화.

### 정리 3.5 — Logit vs Probit vs Log-log

**명제**: Bernoulli의 비교 가능한 link function:

| Link | $g(\mu)$ | $g^{-1}(\eta)$ | Canonical? | 해석 |
|------|----------|----------------|-----------|------|
| Logit | $\log\frac{\mu}{1-\mu}$ | $\sigma(\eta)$ | ✅ | Odds ratio |
| Probit | $\Phi^{-1}(\mu)$ | $\Phi(\eta)$ | ❌ | Latent Gaussian threshold |
| Cloglog | $\log(-\log(1-\mu))$ | $1 - e^{-e^\eta}$ | ❌ | Extreme value survival |

**Probit vs Logit**: 근접. $\beta_{\text{logit}} \approx 1.7 \cdot \beta_{\text{probit}}$ (두 분포의 quantile 함수의 기울기 비).

**Cloglog**: 비대칭, 큰 $\eta$에서 $1$로 빠르게 접근, 작은 $\eta$에서 0에 천천히 — survival 분석에서 자연스러움.

### 정리 3.6 — Poisson 회귀의 IRLS

**명제**: Poisson 회귀 ($\mu = e^\eta$, log link = canonical). Score: $X^\top (y - \mu)$. Hessian: $-X^\top \text{diag}(\mu) X$. IRLS working response: $z = \eta + (y - \mu)/\mu$.

**증명**: 정리 3.2, 3.3의 Poisson 대입. $A(\theta) = e^\theta$, $A'(\theta) = e^\theta = \mu$, $A''(\theta) = e^\theta = \mu$. $\square$

> 💡 **Gaussian 회귀와의 통일**: Normal에서 $A''(\theta) = \sigma^2$ (상수) → $W = (1/\sigma^2) I$ → IRLS 한 iteration에 수렴 (상수 가중치 OLS). Canonical link GLM에서 Normal이 "가장 간단한 특수 사례".

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.linear_model import PoissonRegressor, LogisticRegression
import statsmodels.api as sm

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. GLM IRLS 일반화 함수 — canonical link 가정
# ─────────────────────────────────────────────
def glm_irls(X, y, family='binomial', n_iter=50, tol=1e-10):
    n, p = X.shape
    beta = np.zeros(p)
    
    # 분포별 mu = A'(eta), W = A''(eta)
    if family == 'binomial':
        mu_fn  = lambda eta: 1.0 / (1.0 + np.exp(-eta))
        var_fn = lambda mu:  mu * (1 - mu)
    elif family == 'poisson':
        mu_fn  = lambda eta: np.exp(eta)
        var_fn = lambda mu:  mu
    elif family == 'gaussian':
        mu_fn  = lambda eta: eta
        var_fn = lambda mu:  np.ones_like(mu)
    
    for it in range(n_iter):
        eta = X @ beta
        mu  = mu_fn(eta)
        W   = var_fn(mu) + 1e-12
        z   = eta + (y - mu) / W
        A   = X.T @ (W[:, None] * X)
        b   = X.T @ (W * z)
        beta_new = np.linalg.solve(A, b)
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    return beta, it + 1

# ─────────────────────────────────────────────
# 2. Bernoulli (Logistic Regression)
# ─────────────────────────────────────────────
n, p = 500, 4
X = rng.standard_normal((n, p))
beta_true_log = np.array([1.0, -1.5, 0.5, 0.3])
y_log = (rng.uniform(size=n) < 1/(1 + np.exp(-X @ beta_true_log))).astype(int)

beta_glm, iters = glm_irls(X, y_log, 'binomial')
sk = LogisticRegression(fit_intercept=False, C=1e10, max_iter=2000, tol=1e-12).fit(X, y_log)
print(f'Binomial (Logit):')
print(f'  GLM IRLS : {beta_glm}')
print(f'  sklearn  : {sk.coef_[0]}')
print(f'  차이     : {np.linalg.norm(beta_glm - sk.coef_[0]):.2e}')

# ─────────────────────────────────────────────
# 3. Poisson (log link — canonical)
# ─────────────────────────────────────────────
beta_true_pois = np.array([0.5, -0.3, 0.2, 0.0])
mu_pois = np.exp(X @ beta_true_pois)
y_pois = rng.poisson(mu_pois)

beta_pois, iters = glm_irls(X, y_pois, 'poisson')
sm_pois = sm.GLM(y_pois, X, family=sm.families.Poisson()).fit()
print(f'\nPoisson (Log link — canonical):')
print(f'  True     : {beta_true_pois}')
print(f'  GLM IRLS : {beta_pois}')
print(f'  statsmodels: {sm_pois.params}')
print(f'  차이     : {np.linalg.norm(beta_pois - sm_pois.params):.2e}')

# ─────────────────────────────────────────────
# 4. Gaussian (identity link — canonical) = OLS
# ─────────────────────────────────────────────
y_norm = X @ beta_true_log + 0.3 * rng.standard_normal(n)
beta_norm, iters = glm_irls(X, y_norm, 'gaussian')
beta_ols = np.linalg.solve(X.T @ X, X.T @ y_norm)
print(f'\nGaussian (Identity link — canonical):')
print(f'  GLM IRLS (1 iter): {beta_norm}')
print(f'  OLS (direct)     : {beta_ols}')
print(f'  차이             : {np.linalg.norm(beta_norm - beta_ols):.2e}')
print(f'  (iter 수: {iters}) — 정규분포는 1 step에 수렴')

# ─────────────────────────────────────────────
# 5. Logit vs Probit 비교 (정리 3.5)
# ─────────────────────────────────────────────
from scipy.stats import norm

# Probit IRLS — non-canonical, 약간 다른 form
def probit_nr(X, y, n_iter=100, tol=1e-8):
    p = X.shape[1]
    beta = np.zeros(p)
    for it in range(n_iter):
        eta = X @ beta
        mu = norm.cdf(eta)
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        phi = norm.pdf(eta)
        # Gradient (non-canonical)
        g = X.T @ (phi * (y - mu) / (mu * (1 - mu)))
        # Hessian (observed info)
        w = phi**2 / (mu * (1 - mu))
        H = -X.T @ (w[:, None] * X)
        update = np.linalg.solve(-H, g)
        beta_new = beta + update
        if np.linalg.norm(update) < tol:
            break
        beta = beta_new
    return beta

beta_probit = probit_nr(X, y_log)
print(f'\nLogit vs Probit 계수 비교:')
print(f'  Logit  β̂ : {beta_glm}')
print(f'  Probit β̂ : {beta_probit}')
print(f'  Logit/Probit ratio: {beta_glm / beta_probit}  (이론값 ≈ 1.7)')
```

**출력 예시**:
```
Binomial (Logit):
  GLM IRLS : [ 1.018 -1.524  0.523  0.321]
  sklearn  : [ 1.018 -1.524  0.523  0.321]
  차이     : 1.43e-12

Poisson (Log link — canonical):
  True     : [ 0.5 -0.3  0.2  0. ]
  GLM IRLS : [ 0.498 -0.306  0.205 -0.011]
  statsmodels: [ 0.498 -0.306  0.205 -0.011]
  차이     : 4.21e-10

Gaussian (Identity link — canonical):
  GLM IRLS (1 iter): [ 1.018 -1.542  0.502  0.308]
  OLS (direct)     : [ 1.018 -1.542  0.502  0.308]
  차이             : 4.21e-15
  (iter 수: 2) — 정규분포는 1 step에 수렴

Logit vs Probit 계수 비교:
  Logit  β̂ : [ 1.018 -1.524  0.523  0.321]
  Probit β̂ : [ 0.601 -0.902  0.312  0.191]
  Logit/Probit ratio: [1.69 1.69 1.68 1.68]  (이론값 ≈ 1.7)
```

---

## 🔗 실전 활용

- **Poisson Regression**: count data (사고 수, click-through). `statsmodels.GLM(family=Poisson)`.
- **Gamma Regression**: 양수 연속 데이터, 보험 수가·대기 시간. log link 일반적.
- **Negative Binomial**: Poisson의 overdispersion 대응.
- **Quasi-likelihood**: 분포 명시 없이 mean-variance 관계만 가정. McCullagh.
- **Tweedie**: 연속+0 혼합 (보험 청구액), Poisson과 Gamma 사이.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| 지수족 가정 | Heavy-tailed·multimodal·skewed 비표준 분포는 GLM이 부적합 |
| 선형 $\eta$ | 비선형 효과는 spline, GAM |
| 독립 관측 | 시계열·패널에는 GEE (Generalized Estimating Equations) |
| Canonical link 의존 | Canonical이 아니면 Fisher scoring ≠ Newton — 알고리즘 복잡 |

---

## 📌 핵심 정리

$$\boxed{\text{GLM with canonical link:} \quad \nabla \ell = X^\top(y - \mu),\ \nabla^2 \ell = -X^\top W X,\ W = A''(\eta) = \text{Var}(y)}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **Exponential Family** | $p(y;\theta) = h(y)\exp(\theta^\top T(y) - A(\theta))$ |
| **GLM 3요소** | 분포 + $\eta = x^\top \beta$ + link $g(\mu) = \eta$ |
| **Canonical link** | $g = (A')^{-1}$ → $\eta = \theta$ |
| **Score form** | $X^\top(y - \mu)$ — OLS와 같은 잔차 구조 |
| **IRLS = Fisher scoring** | Canonical이면 Newton과 일치 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Binomial 분포 ($n$회 시행 중 성공 수)의 $A(\theta)$와 canonical link를 유도하라.

<details>
<summary>힌트 및 해설</summary>

$p(y; p_s) = \binom{n}{y} p_s^y (1-p_s)^{n-y}$. $\theta = \log\frac{p_s}{1-p_s}$로 reparametrize → $p_s = \sigma(\theta)$. 

$p(y; \theta) = \binom{n}{y} \exp\bigl(y \theta - n \log(1 + e^\theta)\bigr)$. 

$A(\theta) = n \log(1 + e^\theta)$, $T(y) = y$.

$\mu = A'(\theta) = n \sigma(\theta)$. Canonical link: $\theta = \log\frac{\mu/n}{1 - \mu/n} = \text{logit}(\mu/n)$.

</details>

**문제 2** (심화): Gamma 분포 (shape $k$ 알려짐, rate $\lambda$가 unknown)의 GLM을 유도하라.

<details>
<summary>힌트 및 해설</summary>

$p(y; \lambda) = \frac{\lambda^k}{\Gamma(k)} y^{k-1} e^{-\lambda y}$. $\theta = -\lambda$, $T(y) = y$, $A(\theta) = -k \log(-\theta)$ (with base measure의 shape 부분 별도). 

$\mu = A'(\theta) = -k/\theta = k/\lambda$. Canonical link: $\theta = -k/\mu$, 즉 $g(\mu) = -k/\mu$. 실무에서는 부호를 빼고 $g(\mu) = 1/\mu$ (inverse link).

**주의**: Gamma 회귀의 더 흔한 link는 **log** ($\log \mu = \eta$) — canonical이 아니지만 해석 쉽고 수치 안정. R의 `glm(..., family=Gamma(link='log'))` 기본값.

</details>

**문제 3** (ML 연결): NN에서 출력층의 활성함수(sigmoid·softmax·linear·exp)를 분포와 어떻게 매칭시켜야 하는가?

<details>
<summary>힌트 및 해설</summary>

| 문제 타입 | 활성 함수 | 손실 | 대응 GLM |
|----------|----------|------|---------|
| 회귀 (continuous) | linear | MSE | Normal + identity |
| 이진 분류 | sigmoid | BCE | Binomial + logit |
| 다중 분류 | softmax | CE | Multinomial + logit |
| Count 예측 | exp | Poisson NLL | Poisson + log |
| 양수 continuous | softplus / exp | Gamma NLL | Gamma + log |

**즉 NN의 출력층 + 손실 = GLM + canonical link**. NN은 "GLM에서 선형 $\eta = x^\top \beta$를 비선형 $\eta = f_\theta(x)$로 대체한 것". GLM 이해는 NN 출력층 설계의 기반.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. IRLS](./02-irls.md) | [📚 README](../README.md) | [04. Multinomial·Softmax Regression ▶](./04-softmax-multinomial.md) |

</div>
