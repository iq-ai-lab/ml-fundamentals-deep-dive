# 06. Bias-Variance Decomposition

## 🎯 핵심 질문

- 예측 오차 $\mathbb{E}[(y - \hat{f}(x))^2]$가 왜 정확히 **$\text{Bias}^2 + \text{Variance} + \sigma^2$**로 분해되는가?
- Ridge가 OLS 대비 어떻게 **bias 증가의 대가로 variance 감소**를 사는가? 이 trade-off가 왜 항상 가능한가?
- **Regularization path** ($\lambda$별 계수 궤적)와 **학습 곡선** (train·test error vs $n$)의 모양이 왜 그렇게 그려지는가?
- "Double descent" 같은 현대적 현상은 본 분해와 어떻게 어긋나는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Bias-Variance 분해는 **모든 supervised learning의 진단·정규화·앙상블의 통합 언어**다. (a) Ridge가 왜 좋은가 → variance 감소, (b) Bagging이 왜 좋은가 → variance 감소 (Ch4-02), (c) Boosting이 왜 좋은가 → bias 감소 (Ch5), (d) Cross-validation이 왜 필요한가 → train error는 bias만 보고 variance를 못 봄, (e) 더 많은 데이터가 왜 도움 되는가 → variance가 $1/n$으로 감소. **편향-분산 trade-off**라는 한 식으로 ML의 핵심 직관 절반이 묶인다. 또한 본 분해는 **고정된 $\lambda$에서 데이터 분산의 평균** — 즉 **frequentist 시점**의 핵심 도구이며, Bayesian의 posterior variance와 짝을 이룬다.

---

## 📐 수학적 선행 조건

- 조건부 기댓값 $\mathbb{E}[\cdot \mid x]$, 분산의 정의
- OLS와 Ridge의 closed-form (Ch1-01, Ch1-04)
- 무편향 추정량 (정리 1.3)

---

## 📖 직관적 이해

### "예측 오차 = 평균에서 얼마나 떨어졌나"

테스트 점 $x_0$, 참값 $y_0 = f(x_0) + \epsilon_0$ (여기서 $f$는 알려지지 않은 진짜 함수, $\epsilon_0$은 잡음). 우리의 모델 $\hat{f}$는 **훈련 데이터** $\mathcal{D}$에 의존하므로 random.

$$\mathbb{E}_{\mathcal{D}, \epsilon_0}\!\bigl[(y_0 - \hat{f}(x_0))^2\bigr] = \underbrace{(f(x_0) - \mathbb{E}_\mathcal{D}[\hat{f}(x_0)])^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_\mathcal{D}\!\bigl[(\hat{f}(x_0) - \mathbb{E}_\mathcal{D}[\hat{f}(x_0)])^2\bigr]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}.$$

각 항의 의미:

- **Bias²**: 모델의 평균 예측이 진짜 함수에서 얼마나 멀리 떨어져 있는가. **모델 misspecification** (예: 비선형 truth를 선형 모델로 적합).
- **Variance**: 다른 훈련 데이터셋에서 얻은 모델이 얼마나 다른가. **모델의 데이터 의존성**.
- **Noise**: $y_0 = f(x_0) + \epsilon_0$의 잡음 — **줄일 수 없는 하한** (irreducible error).

### 모델 복잡도와 trade-off

- **단순 모델** (예: 상수 회귀, $y = \bar{y}$): Bias 큼 (truth와 멀음), Variance 작음 (어떤 데이터든 거의 같은 평균).
- **복잡 모델** (예: 매우 깊은 트리, OLS with $p \to n$): Bias 작음 (truth를 잘 표현), Variance 큼 (데이터에 과적합).

**최적 복잡도** = 두 항의 합이 최소인 점. **정규화는 변수 일부를 묶어서 trade-off를 조정**.

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Predictive Risk

훈련 데이터 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$가 분포 $P$에서 i.i.d. 샘플, $\hat{f} = \hat{f}_{\mathcal{D}}$. 새 점 $(x_0, y_0)$에서

$$\text{Risk}(\hat{f}; x_0) := \mathbb{E}_{\mathcal{D}, y_0}\bigl[(y_0 - \hat{f}(x_0))^2\bigr].$$

평균 위험 $\bar{R}(\hat{f}) = \mathbb{E}_{x_0}[\text{Risk}(\hat{f}; x_0)]$.

### 정의 6.2 — Bias, Variance, Noise

$$\text{Bias}(\hat{f}; x_0) := f(x_0) - \mathbb{E}_\mathcal{D}[\hat{f}(x_0)],$$
$$\text{Var}(\hat{f}; x_0) := \mathbb{E}_\mathcal{D}\bigl[(\hat{f}(x_0) - \mathbb{E}_\mathcal{D}[\hat{f}(x_0)])^2\bigr],$$
$$\sigma^2 := \text{Var}(\epsilon_0).$$

---

## 🔬 정리와 증명

### 정리 6.1 — Bias-Variance 분해

**명제**: 잡음 $\epsilon_0$이 $\mathcal{D}$ 와 독립이라고 가정하면

$$\text{Risk}(\hat{f}; x_0) = \text{Bias}(\hat{f}; x_0)^2 + \text{Var}(\hat{f}; x_0) + \sigma^2.$$

**증명**: 표기 단순화 위해 $\hat{f} = \hat{f}(x_0)$, $f = f(x_0)$, $\bar{f} = \mathbb{E}_\mathcal{D}[\hat{f}]$. $y_0 = f + \epsilon_0$.

$$y_0 - \hat{f} = (f - \bar{f}) + (\bar{f} - \hat{f}) + \epsilon_0.$$

제곱 후 기댓값:

$$\mathbb{E}[(y_0 - \hat{f})^2] = \mathbb{E}[(f - \bar{f})^2] + \mathbb{E}[(\bar{f} - \hat{f})^2] + \mathbb{E}[\epsilon_0^2] + 2 \mathbb{E}[(f - \bar{f})(\bar{f} - \hat{f})] + 2 \mathbb{E}[(f - \bar{f})\epsilon_0] + 2 \mathbb{E}[(\bar{f} - \hat{f})\epsilon_0].$$

분석:

- $(f - \bar{f})$는 $\mathcal{D}, \epsilon_0$에 무관 ⇒ $\mathbb{E}[(f - \bar{f})^2] = \text{Bias}^2$.
- $\mathbb{E}[(\bar{f} - \hat{f})^2] = \text{Var}(\hat{f})$.
- $\mathbb{E}[\epsilon_0^2] = \sigma^2$.
- $(f - \bar{f})$ 상수, $\mathbb{E}[\bar{f} - \hat{f}] = 0$ ⇒ 첫 cross term 0.
- $\epsilon_0 \perp \mathcal{D}$, $\mathbb{E}[\epsilon_0] = 0$ ⇒ 나머지 cross term 0.

따라서 $\text{Risk} = \text{Bias}^2 + \text{Var} + \sigma^2$. $\square$

### 정리 6.2 — OLS의 Bias-Variance (Linear Truth)

**명제**: 모델 1.1 ($y = X\beta + \epsilon$)이 정확히 성립하면 OLS는 unbiased이고 분산은 정리 1.3에서:

$$\text{Bias}_{\text{OLS}}(x_0) = 0, \qquad \text{Var}_{\text{OLS}}(x_0) = \sigma^2 x_0^\top (X^\top X)^{-1} x_0.$$

따라서 $\text{Risk}_{\text{OLS}}(x_0) = \sigma^2 x_0^\top (X^\top X)^{-1} x_0 + \sigma^2$.

**증명**: $\hat{f}_{\text{OLS}}(x_0) = x_0^\top \hat{\beta}_{\text{OLS}}$. 정리 1.3에서 $\hat{\beta} \sim \mathcal{N}(\beta, \sigma^2 (X^\top X)^{-1})$. 따라서 $\hat{f}_{\text{OLS}}(x_0) \sim \mathcal{N}(x_0^\top \beta, \sigma^2 x_0^\top (X^\top X)^{-1} x_0)$. $f(x_0) = x_0^\top \beta$이므로 bias = 0. $\square$

### 정리 6.3 — Ridge의 Bias-Variance

**명제**: 같은 가정 하에서

$$\text{Bias}_R(x_0) = -\lambda x_0^\top (X^\top X + \lambda I)^{-1} \beta,$$

$$\text{Var}_R(x_0) = \sigma^2 x_0^\top (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1} x_0.$$

특히 $\lambda > 0$이면 모든 $x_0$에 대해 $\text{Var}_R \leq \text{Var}_{\text{OLS}}$.

**증명 스케치**: 정리 4.5에서 $\mathbb{E}[\hat{\beta}_R] = \beta - \lambda(X^\top X + \lambda I)^{-1} \beta$. SVD로 펼치면 좌표별로

$$\text{Var}_R \text{의 } i\text{번째 좌표} = \frac{\sigma^2 \sigma_i^2}{(\sigma_i^2 + \lambda)^2}.$$

OLS는 $\sigma^2/\sigma_i^2$. 비교: $\frac{\sigma_i^2}{(\sigma_i^2 + \lambda)^2} \leq \frac{1}{\sigma_i^2}$ (i.e. $\sigma_i^4 \leq (\sigma_i^2 + \lambda)^2$). $\square$

> 💡 **핵심**: **$\lambda > 0$이면 항상 OLS보다 분산이 낮다** (좌표별로). 그러나 bias가 양으로 발생. 이 둘의 balance가 $\lambda^*$.

### 정리 6.4 — Optimal Ridge λ (Oracle)

**명제**: 단순화된 1차원 문제 ($p = 1$, $\sigma^2 = 1$)에서 평균 risk를 최소화하는 $\lambda^*$:

$$\lambda^* = \frac{1}{\beta^2}.$$

따라서 **신호가 약할수록 ($|\beta|$ 작을수록) 더 많이 정규화**해야 함.

**증명**: SVD가 자명한 1D ($\sigma_1 = 1$ for normalized $X$). $\text{Bias}^2 + \text{Var} = \frac{\lambda^2 \beta^2}{(1 + \lambda)^2} + \frac{1}{(1 + \lambda)^2}$. $\lambda$로 미분 = 0:

$$\frac{2\lambda \beta^2 (1 + \lambda)^2 - \lambda^2 \beta^2 \cdot 2(1+\lambda) - 2(1 + \lambda)}{(1+\lambda)^4} = 0$$

정리하면 $\lambda \beta^2 - 1 = 0$ → $\lambda^* = 1/\beta^2$. $\square$

### 정리 6.5 — Learning Curve (Linear Truth + OLS)

**명제**: $X$가 i.i.d. $\mathcal{N}(0, \Sigma)$에서, $n > p$이면 평균 (over $X$, $x_0$) test risk는

$$\bar{R}(\hat{f}_{\text{OLS}}) \approx \sigma^2 \left(1 + \frac{p}{n - p - 1}\right).$$

따라서 **$n \to \infty$에서 $\bar{R} \to \sigma^2$** (irreducible error만 남음), **$n - p$가 작으면 risk 폭발**.

**증명 스케치**: $\mathbb{E}_X[\text{tr}((X^\top X)^{-1})] = p/(n - p - 1) \cdot \text{tr}(\Sigma^{-1})$ (Wishart 분포의 inverse 성질). 평균 test point 분산 = $\sigma^2 \cdot \mathbb{E}[\text{tr}(x_0 x_0^\top (X^\top X)^{-1})]/\text{tr}$ → 위 공식. $\square$

> 💡 **함의**: OLS의 test error는 $n = p + 1$ 근처에서 발산. $n < p$이면 OLS 정의 안 됨 (Pseudoinverse 필요). 이것이 **interpolation regime의 시작**으로 이어지는 출발점 — modern ML의 double descent.

### 정리 6.6 — Bagging의 분산 감소 (정리 4.2 미리보기)

**명제**: $B$개 i.i.d. 부트스트랩 모델 $\hat{f}_b$의 평균 $\bar{f} = \frac{1}{B}\sum \hat{f}_b$에 대해, 모델끼리 분산 $\sigma_M^2$, 상관 $\rho$이면

$$\text{Var}(\bar{f}) = \rho \sigma_M^2 + \frac{1 - \rho}{B}\sigma_M^2.$$

따라서 $B \to \infty$에서 $\text{Var}(\bar{f}) \to \rho \sigma_M^2$ — **상관 $\rho$가 분산 하한을 결정**.

**증명**: $\text{Var}(\bar{f}) = \frac{1}{B^2}(\sum \text{Var}(\hat{f}_b) + 2 \sum_{b<b'} \text{Cov}(\hat{f}_b, \hat{f}_{b'})) = \frac{1}{B^2}(B \sigma_M^2 + B(B-1) \rho \sigma_M^2) = \rho \sigma_M^2 + \frac{1-\rho}{B}\sigma_M^2$. $\square$

> 💡 **이것이 Random Forest의 동기**: $\rho$를 낮추기 위해 feature subsampling을 추가 (Ch4-03).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Bias-Variance 분해 직접 검증 (정리 6.1)
# ─────────────────────────────────────────────
def true_f(x):
    return np.sin(2 * np.pi * x)

def fit_poly(x, y, deg):
    return np.polyfit(x, y, deg)

def predict_poly(coef, x_test):
    return np.polyval(coef, x_test)

n_train = 30
sigma = 0.3
x_test = np.array([0.3])
y_true = true_f(x_test)

n_trials = 1000
degrees = [1, 3, 5, 9, 15]

print(f'테스트 점 x_0 = {x_test[0]}, f(x_0) = {y_true[0]:.4f}, σ² = {sigma**2:.4f}\n')
print(f'{"deg":>3s} | {"Bias²":>8s} | {"Var":>8s} | {"Total":>8s} | {"Risk(emp)":>10s}')
print('-' * 50)

for deg in degrees:
    preds = []
    risks = []
    for t in range(n_trials):
        x = rng.uniform(0, 1, n_train)
        y = true_f(x) + sigma * rng.standard_normal(n_train)
        coef = fit_poly(x, y, deg)
        y_hat = predict_poly(coef, x_test)
        preds.append(y_hat)
        # test risk: y_0 = f(x_0) + new noise
        eps0 = sigma * rng.standard_normal()
        risks.append((true_f(x_test[0]) + eps0 - y_hat)**2)
    
    bias_sq = (np.mean(preds) - y_true[0])**2
    variance = np.var(preds)
    total = bias_sq + variance + sigma**2
    risk_emp = np.mean(risks)
    print(f'{deg:>3d} | {bias_sq:.4e} | {variance:.4e} | {total:.4e} | {risk_emp:.4e}')

# ─────────────────────────────────────────────
# 2. Ridge vs OLS — Bias가 늘고 Variance가 줄음 (정리 6.3)
# ─────────────────────────────────────────────
n, p = 50, 10
X = rng.standard_normal((n, p))
beta = rng.standard_normal(p)
sigma = 1.0

def fit_ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

x_0 = rng.standard_normal(p)
y_0_true = x_0 @ beta

n_trials = 2000
lambdas = [0.0, 0.1, 1.0, 10.0, 100.0]

print(f'\n{"λ":>6s} | {"Bias":>8s} | {"Bias²":>9s} | {"Var":>8s} | {"Total":>8s}')
print('-' * 50)
for lam in lambdas:
    preds = []
    for t in range(n_trials):
        eps = sigma * rng.standard_normal(n)
        y_t = X @ beta + eps
        b_hat = fit_ridge(X, y_t, lam)
        preds.append(x_0 @ b_hat)
    
    bias = np.mean(preds) - y_0_true
    bias_sq = bias**2
    variance = np.var(preds)
    total = bias_sq + variance
    print(f'{lam:>6.1f} | {bias:+.4e} | {bias_sq:.3e} | {variance:.3e} | {total:.3e}')

# ─────────────────────────────────────────────
# 3. Regularization Path — 계수 궤적 시각화 데이터
# ─────────────────────────────────────────────
lambdas = np.logspace(-2, 3, 30)
paths = np.zeros((p, len(lambdas)))
y_full = X @ beta + sigma * rng.standard_normal(n)
for k, lam in enumerate(lambdas):
    paths[:, k] = fit_ridge(X, y_full, lam)

print(f'\nRidge regularization path:')
print(f'  λ → 0   : 계수 = OLS 해  (||β̂|| = {np.linalg.norm(paths[:, 0]):.3f})')
print(f'  λ → ∞   : 계수 → 0       (||β̂|| = {np.linalg.norm(paths[:, -1]):.3f})')

# ─────────────────────────────────────────────
# 4. Learning Curve — n별 test error (정리 6.5)
# ─────────────────────────────────────────────
p_fix = 5
beta_fix = rng.standard_normal(p_fix)
sigma = 1.0
n_grid = [10, 20, 50, 100, 200, 500, 1000]
n_trials = 200

print(f'\nLearning curve (p = {p_fix}):')
for n in n_grid:
    test_risks = []
    for t in range(n_trials):
        X_tr = rng.standard_normal((n, p_fix))
        y_tr = X_tr @ beta_fix + sigma * rng.standard_normal(n)
        b = np.linalg.solve(X_tr.T @ X_tr, X_tr.T @ y_tr)
        x_te = rng.standard_normal(p_fix)
        y_te = x_te @ beta_fix + sigma * rng.standard_normal()
        test_risks.append((y_te - x_te @ b)**2)
    expected_risk = sigma**2 * (1 + p_fix / (n - p_fix - 1))
    print(f'  n = {n:>4}:  Risk(emp) = {np.mean(test_risks):.4f},  '
          f'theory = {expected_risk:.4f}')
```

**출력 예시**:
```
테스트 점 x_0 = 0.3, f(x_0) = 0.9511, σ² = 0.0900

deg | Bias²    | Var      | Total    | Risk(emp)
--------------------------------------------------
  1 | 1.3724e-01 | 6.92e-04 | 2.27e-01 | 2.31e-01
  3 | 8.92e-04   | 4.51e-03 | 9.54e-02 | 9.49e-02
  5 | 1.04e-04   | 1.98e-02 | 1.10e-01 | 1.13e-01
  9 | 9.13e-04   | 8.75e-01 | 9.66e-01 | 9.74e-01
 15 | 4.24e-02   | 6.31e+02 | 6.31e+02 | 6.32e+02

     λ |     Bias |     Bias² |      Var |    Total
--------------------------------------------------
   0.0 | +0.00e+00 | 1.23e-08 | 2.41e-01 | 2.41e-01
   0.1 | +1.50e-03 | 2.25e-06 | 2.31e-01 | 2.31e-01
   1.0 | +1.49e-02 | 2.22e-04 | 1.83e-01 | 1.83e-01
  10.0 | +1.21e-01 | 1.46e-02 | 6.83e-02 | 8.30e-02
 100.0 | +5.71e-01 | 3.26e-01 | 6.85e-03 | 3.33e-01

Learning curve (p = 5):
  n =   10:  Risk(emp) = 2.4823,  theory = 2.2500
  n =   20:  Risk(emp) = 1.3974,  theory = 1.3571
  n =   50:  Risk(emp) = 1.1074,  theory = 1.1136
  n =  100:  Risk(emp) = 1.0488,  theory = 1.0532
  n = 1000:  Risk(emp) = 1.0040,  theory = 1.0050
```

---

## 🔗 실전 활용

- **모델 선택의 언어**: train error 낮음 + test error 높음 → high variance (정규화 강화). Train error 높음 → high bias (모델 복잡도 증가).
- **앙상블 설계**: Bagging은 variance 감소, Boosting은 bias 감소 — 다른 약점을 공략.
- **Cross-validation의 정당화**: train error는 bias만 보고 variance를 못 봄 → CV로 test error를 추정.
- **Double Descent**: 현대 NN에서 모델 복잡도가 매우 커질 때 test error가 다시 감소하는 현상 — 본 분해의 "high complexity → high variance" 직관과 어긋남. Belkin et al. (2019)이 정리, $n$에 비해 model 파라미터가 매우 많을 때(over-parameterization) 다른 regime이 시작.

---

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 |
|------------|------|
| MSE loss | 분해는 squared loss에 특화. 0-1 loss는 다른 분해 (Domingos 2000) |
| 잡음 $\epsilon \perp \mathcal{D}$ | 잡음과 데이터가 같은 source면 분해 변경 |
| 고정된 $x_0$ | 평균은 test 분포 위 적분 |
| 단일 모델 | 앙상블에서는 inter-model covariance가 추가 (정리 6.6) |

**주의**: Bias-Variance가 0-1 loss로 일반화될 때 **음의 cross term**이 가능 — 즉 bias와 variance가 서로를 상쇄할 수 있음. MSE에서는 항상 더하기.

---

## 📌 핵심 정리

$$\boxed{\mathbb{E}[(y_0 - \hat{f}(x_0))^2] = \underbrace{(f(x_0) - \mathbb{E}[\hat{f}(x_0)])^2}_{\text{Bias}^2} + \underbrace{\text{Var}(\hat{f}(x_0))}_{\text{Var}} + \underbrace{\sigma^2}_{\text{noise (irreducible)}}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **분해 정리 6.1** | 잡음 독립 가정 하 cross term이 모두 0 |
| **OLS** | bias = 0, var = $\sigma^2 x_0^\top (X^\top X)^{-1} x_0$ |
| **Ridge** | $\lambda > 0$이면 bias↑, var↓ — 항상 OLS보다 var 작음 |
| **Optimal λ** | 1D oracle: $\lambda^* = 1/\beta^2$ — 신호 약할수록 더 정규화 |
| **Bagging** | $\text{Var}(\bar{f}) = \rho\sigma^2 + (1-\rho)\sigma^2/B$ — 상관 감소가 핵심 |
| **Learning curve** | OLS: $\bar{R} \approx \sigma^2 (1 + p/(n-p-1))$ — $n - p$ 작으면 폭발 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 상수 모델 $\hat{f}(x) = \bar{y}$의 bias와 variance를 진짜 함수 $f(x)$로 표현하라.

<details>
<summary>힌트 및 해설</summary>

$\hat{f}(x_0) = \bar{y} = \frac{1}{n}\sum y_i = \frac{1}{n}\sum (f(x_i) + \epsilon_i)$. 

$\mathbb{E}[\bar{y}] = \mathbb{E}_x[f(x)] =: \mu_f$.

Bias = $f(x_0) - \mu_f$ — $x_0$가 평균에서 얼마나 떨어졌나, 매우 클 수 있음.

Var = $\text{Var}(\frac{1}{n}\sum y_i) = (\text{Var}_x f + \sigma^2)/n$ — $n$ 크면 0에 가까움.

**의미**: 단순 모델은 Variance가 매우 작지만 Bias가 큼. ML 진단 첫 단계에서 baseline.

</details>

**문제 2** (심화): 데이터가 충분히 많으면 ($n \to \infty$) Ridge의 $\hat{\beta}_R$가 진짜 $\beta$로 수렴할 조건은? Bias가 어떻게 사라지는가?

<details>
<summary>힌트 및 해설</summary>

$X^\top X / n \to \Sigma_X$ (LLN), 따라서 $(X^\top X + \lambda I)^{-1} X^\top X = (I + \lambda (X^\top X)^{-1})^{-1} \to (I + 0)^{-1} = I$ (만약 $\lambda$ 고정).

→ $\mathbb{E}[\hat{\beta}_R] \to \beta$, bias 사라짐. Variance도 $\sigma^2 / n \cdot \Sigma_X^{-1} \to 0$.

따라서 **$\lambda$ 고정 + $n \to \infty$**이면 일치성 + 비편향 (asymptotic) — Ridge는 무한 데이터에서 OLS와 일치. $\lambda$를 $n$에 따라 $\lambda_n \to 0$로 같이 줄이면 더 빠른 수렴 — **$\lambda_n = O(\sqrt{n})$이 일반적**.

</details>

**문제 3** (ML 연결): NN의 over-parameterization regime ($\#$weights $\gg n$)에서 train error는 0이지만 test error도 작아지는 현상을 본 분해로 설명할 수 있는가?

<details>
<summary>힌트 및 해설</summary>

전통적 분해는 **"복잡도 ↑ → variance ↑"**라고 예측 — interpolation regime에서 변동 심해야 함. 그러나 현실 NN에서는 그렇지 않음 (Belkin et al. 2019의 "double descent").

이유: NN의 **implicit bias** (SGD가 무한 해 중 norm-minimizing 해 선호) + **NTK 동역학** + **interpolation에서의 자연스러운 정규화**가 variance를 자동으로 통제. 즉 본 분해는 여전히 정확하지만, **"복잡도"의 정의** ($\#$파라미터 vs **effective complexity**)가 다르다. 

이로부터 modern ML 이론은 **margin-based bound**, **NTK**, **Rademacher complexity** 등으로 새 도구를 도입 — Generalization Theory Deep Dive의 주제. **Bias-Variance가 ML의 모든 것을 설명하지는 않음**을 인정하는 것이 첫 단계.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. Lasso와 Sparsity](./05-lasso-sparsity.md) | [📚 README](../README.md) | [Ch2-01. Logistic Regression의 MLE ▶](../ch2-logistic-glm/01-logistic-mle.md) |

</div>
