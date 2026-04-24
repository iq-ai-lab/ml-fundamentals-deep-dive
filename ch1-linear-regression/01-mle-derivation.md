# 01. MLE 관점에서의 선형 회귀

## 🎯 핵심 질문

- 가정 $y = X\beta + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ 하에서 log-likelihood를 최대화하면 왜 정확히 **최소제곱**이 되는가?
- Normal Equation $\hat{\beta} = (X^\top X)^{-1} X^\top y$는 어떻게 한 줄씩 유도되는가?
- MLE 추정량 $\hat{\beta}$의 분포는? 비편향성·일치성·효율성은 어디서 오는가?
- **Gauss-Markov 정리**(BLUE)는 정규성 가정 없이도 OLS가 "최선"임을 어떻게 보장하는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

선형 회귀는 모든 supervised learning의 **원형(prototype)** 이다. Logistic regression의 GLM 일반화, Ridge/Lasso의 정규화, Kernel Ridge의 RKHS 확장, 신경망의 마지막 layer까지 — 모두 이 한 식 $\hat{\beta} = (X^\top X)^{-1} X^\top y$의 변주다. 그런데 sklearn 한 줄 `LinearRegression().fit(X, y)` 뒤에는 다음이 숨어 있다: **(1) "잡음이 Gaussian"이라는 가정이 있어야 MLE가 최소제곱과 일치**하고, (2) **$X^\top X$가 가역**이어야 closed-form이 존재하며, (3) **잡음의 등분산성·비상관성**이 있어야 OLS가 BLUE가 된다. 이 세 가정이 깨질 때 어떤 식이 어떻게 변하는지 — Pseudoinverse(rank-deficient), GLS(상관 잡음), Ridge(다공선성) — 가 다음 5개 문서의 주제다. 즉 본 문서는 **나머지 모든 회귀법의 베이스라인**이다.

---

## 📐 수학적 선행 조건

- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 행렬 미분, 양정치 행렬, 가역성·rank
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 다변수정규분포, 조건부 기댓값
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): MLE, Fisher 정보, Cramér-Rao 하한
- 미적분학: 기울기 = 0의 1계 조건, Hessian의 양정치성에 의한 2계 조건

---

## 📖 직관적 이해

### "잡음이 Gaussian"이 왜 최소제곱을 부르는가

데이터 생성 모델 $y_i = x_i^\top \beta + \epsilon_i$, $\epsilon_i \stackrel{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$를 받았다. 한 점의 likelihood는

$$p(y_i \mid x_i, \beta) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left( -\frac{(y_i - x_i^\top \beta)^2}{2\sigma^2} \right).$$

전체 log-likelihood는 더하기:

$$\ell(\beta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - x_i^\top \beta)^2.$$

$\beta$에 대해 **상수항을 빼면** 정확히 $-\frac{1}{2\sigma^2}\|y - X\beta\|^2$가 남는다. 따라서 **log-likelihood 최대화 ⇔ $\|y - X\beta\|^2$ 최소화**. 즉 "최소제곱법"은 통계적 추정 원리(MLE)에서 자동으로 나오는 것이지, 어디서 천재적으로 가정된 식이 아니다.

### 미분 = 0 한 줄

$$\nabla_\beta \|y - X\beta\|^2 = -2 X^\top (y - X\beta) = 0 \implies X^\top X \beta = X^\top y.$$

이 식 $X^\top X \beta = X^\top y$가 **정규방정식(Normal Equation)** 이고, $X^\top X$가 가역이면 $\hat{\beta} = (X^\top X)^{-1} X^\top y$.

### "정규방정식"이라는 이름의 의미

"Normal"은 **수직(perpendicular)** 이라는 뜻이다. $X^\top (y - X\hat{\beta}) = 0$은 **잔차 $y - X\hat{\beta}$가 $X$의 모든 열에 수직**이라는 뜻이다. 이 기하학적 의미는 Ch1-02에서 자세히 다룬다.

---

## ✏️ 엄밀한 정의

### 모델 1.1 — Gaussian 선형 모델

데이터 $\{(x_i, y_i)\}_{i=1}^n$, $x_i \in \mathbb{R}^p$, $y_i \in \mathbb{R}$가 다음을 만족한다고 가정한다:

$$y = X\beta + \epsilon, \qquad X \in \mathbb{R}^{n \times p}, \quad \beta \in \mathbb{R}^p, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I_n).$$

여기서 $X$의 $i$번째 행은 $x_i^\top$이다. $\beta$와 $\sigma^2$는 미지의 모수.

### 정의 1.2 — 잔차 제곱합 (RSS)과 OLS 추정량

목적함수

$$\mathrm{RSS}(\beta) := \|y - X\beta\|^2 = \sum_{i=1}^n (y_i - x_i^\top \beta)^2$$

를 **잔차 제곱합** (Residual Sum of Squares)이라 한다. 이를 최소화하는 $\hat{\beta}_{\text{OLS}} = \arg\min_\beta \mathrm{RSS}(\beta)$를 **Ordinary Least Squares (OLS) 추정량**이라 한다.

### 정의 1.3 — Best Linear Unbiased Estimator (BLUE)

선형 추정량 $\tilde{\beta} = Ay$ ($A$는 $\beta$에 의존하지 않는 $p \times n$ 행렬)가 $\mathbb{E}[\tilde{\beta}] = \beta$ (비편향)이고, 임의의 다른 비편향 선형 추정량 $\tilde{\beta}'$에 대해 $\text{Var}(\tilde{\beta}') - \text{Var}(\tilde{\beta}) \succeq 0$ (PSD 의미로 분산이 더 작거나 같음)일 때 $\tilde{\beta}$를 **BLUE**라 한다.

---

## 🔬 정리와 증명

### 정리 1.1 — MLE = OLS

**명제**: 모델 1.1 하에서 $\beta$의 MLE는 OLS 추정량과 일치한다:

$$\hat{\beta}_{\text{MLE}} = \arg\max_\beta \ell(\beta) = \arg\min_\beta \|y - X\beta\|^2 = \hat{\beta}_{\text{OLS}}.$$

**증명**: Gaussian 밀도에서 likelihood는

$$L(\beta, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - x_i^\top \beta)^2}{2\sigma^2}\right) = (2\pi\sigma^2)^{-n/2} \exp\!\left(-\frac{\|y - X\beta\|^2}{2\sigma^2}\right).$$

로그를 취하면

$$\ell(\beta, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|y - X\beta\|^2.$$

$\beta$에 대한 부분만 남기면 $-\frac{1}{2\sigma^2}\|y - X\beta\|^2$. $\sigma^2 > 0$이므로 이를 최대화 = $\|y - X\beta\|^2$ 최소화. $\square$

### 정리 1.2 — Normal Equation의 유도와 최적성

**명제**: $X \in \mathbb{R}^{n \times p}$가 full column rank ($\text{rank}(X) = p$)이면 $\mathrm{RSS}(\beta)$의 유일한 최소화 점은

$$\hat{\beta}_{\text{OLS}} = (X^\top X)^{-1} X^\top y.$$

**증명**:

(1) **1계 조건.** $\mathrm{RSS}(\beta) = (y - X\beta)^\top (y - X\beta) = y^\top y - 2 \beta^\top X^\top y + \beta^\top X^\top X \beta$. 행렬 미분 공식 $\nabla_\beta (\beta^\top A \beta) = (A + A^\top)\beta = 2A\beta$ ($A = X^\top X$ 대칭)와 $\nabla_\beta (b^\top \beta) = b$ ($b = X^\top y$)를 적용하면

$$\nabla_\beta \mathrm{RSS}(\beta) = -2 X^\top y + 2 X^\top X \beta.$$

이것을 $0$으로 놓으면 **Normal Equation**

$$\boxed{X^\top X \beta = X^\top y.}$$

(2) **유일성.** $X$가 full column rank $\Rightarrow$ $X^\top X \succ 0$ (strictly PD). 실제로 임의의 $0 \neq v \in \mathbb{R}^p$에 대해 $v^\top X^\top X v = \|Xv\|^2 > 0$ (full column rank이므로 $Xv \neq 0$). 따라서 $X^\top X$ 가역, $\hat{\beta}_{\text{OLS}} = (X^\top X)^{-1} X^\top y$.

(3) **2계 조건.** $\nabla^2_\beta \mathrm{RSS}(\beta) = 2 X^\top X \succ 0$ — 강볼록(strictly convex). 따라서 1계 조건의 해가 **전역 유일 최소점**. $\square$

> 💡 **Rank-deficient한 경우**($\text{rank}(X) < p$)는 Ch1-03에서 Moore-Penrose pseudoinverse로 처리한다. 이때는 $\hat{\beta}_{\text{OLS}}$가 **유일하지 않고**, $X^+y$가 그중 **min-norm 해**이다.

### 정리 1.3 — OLS의 분포

**명제**: 모델 1.1 하에서 $\hat{\beta}_{\text{OLS}}$의 표본분포는

$$\hat{\beta}_{\text{OLS}} \sim \mathcal{N}\!\bigl(\beta,\ \sigma^2 (X^\top X)^{-1}\bigr).$$

특히 $\hat{\beta}$는 **비편향**이고 분산은 $\sigma^2 (X^\top X)^{-1}$.

**증명**: $\hat{\beta} = (X^\top X)^{-1} X^\top y = (X^\top X)^{-1} X^\top (X\beta + \epsilon) = \beta + (X^\top X)^{-1} X^\top \epsilon$. 따라서

- $\mathbb{E}[\hat{\beta}] = \beta + (X^\top X)^{-1} X^\top \mathbb{E}[\epsilon] = \beta$.
- $\text{Var}(\hat{\beta}) = (X^\top X)^{-1} X^\top \cdot \sigma^2 I \cdot X (X^\top X)^{-1} = \sigma^2 (X^\top X)^{-1}$.
- Gaussian의 affine 변환은 Gaussian이므로 $\hat{\beta} \sim \mathcal{N}(\beta, \sigma^2 (X^\top X)^{-1})$. $\square$

### 정리 1.4 — Gauss-Markov (BLUE)

**명제**: $\mathbb{E}[\epsilon] = 0$, $\text{Var}(\epsilon) = \sigma^2 I$를 가정한다 (**Gaussian 가정 불필요**). 그러면 OLS 추정량 $\hat{\beta}_{\text{OLS}}$는 $\beta$의 모든 비편향 선형 추정량 가운데 분산이 가장 작다 (BLUE).

**증명**: $\hat{\beta} = Cy$ where $C = (X^\top X)^{-1} X^\top$. 다른 비편향 선형 추정량 $\tilde{\beta} = (C + D)y$가 비편향이려면 $\mathbb{E}[\tilde{\beta}] = (C+D)X\beta = \beta$가 모든 $\beta$에 대해 성립해야 하므로 $(C + D)X = I$. $CX = (X^\top X)^{-1} X^\top X = I$이므로 **$DX = 0$**.

$\tilde{\beta}$의 분산:

$$\text{Var}(\tilde{\beta}) = (C + D) \sigma^2 I (C + D)^\top = \sigma^2 (CC^\top + CD^\top + DC^\top + DD^\top).$$

$CD^\top = (X^\top X)^{-1} X^\top D^\top = (X^\top X)^{-1} (DX)^\top = 0$, $DC^\top = 0$. 따라서

$$\text{Var}(\tilde{\beta}) = \sigma^2 (X^\top X)^{-1} + \sigma^2 DD^\top = \text{Var}(\hat{\beta}) + \sigma^2 DD^\top.$$

$DD^\top \succeq 0$이므로 $\text{Var}(\tilde{\beta}) \succeq \text{Var}(\hat{\beta})$, 등호는 $D = 0$일 때만. $\square$

> 💡 **함의**: Gauss-Markov는 정규성을 요구하지 않는다. 잡음이 등분산이고 비상관이면 OLS가 모든 선형 비편향 추정량 중 최소분산. 그래서 OLS는 **잡음 분포 모델 misspecification에 어느 정도 robust**하다.

### 정리 1.5 — $\sigma^2$의 MLE와 비편향 추정

**명제**: $\sigma^2$의 MLE는 $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\|y - X\hat{\beta}\|^2$이고, 이는 **편향**되어 있다 ($\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-p}{n}\sigma^2$). 비편향 추정량은

$$\hat{\sigma}^2_{\text{unbiased}} = \frac{1}{n - p}\|y - X\hat{\beta}\|^2 = \frac{\mathrm{RSS}}{n - p}.$$

**증명 스케치**: $y - X\hat{\beta} = (I - H)y = (I - H)\epsilon$ ($H = X(X^\top X)^{-1} X^\top$). $\mathbb{E}[\epsilon^\top (I - H) \epsilon] = \sigma^2 \cdot \text{tr}(I - H) = \sigma^2(n - p)$. 따라서 $\hat{\sigma}^2_{\text{MLE}}$는 $\frac{n-p}{n}\sigma^2$로 편향. 자유도 보정 $1/(n-p)$가 비편향을 만든다. $\square$

> 📌 sklearn `LinearRegression()`은 $\hat{\beta}$만 돌려주고 $\sigma^2$는 따로 추정하지 않는다. `statsmodels.OLS().fit()`은 자동으로 $1/(n-p)$ 보정된 $\hat{\sigma}^2$를 보고한다.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. 합성 데이터에서 MLE = OLS 확인
# ─────────────────────────────────────────────
n, p = 200, 5
X = rng.standard_normal((n, p))
beta_true = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
sigma_true = 0.5
y = X @ beta_true + sigma_true * rng.standard_normal(n)

# Normal Equation 직접
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
print(f'OLS  : {beta_hat}')
print(f'True : {beta_true}\n')

# sklearn과 비교
sk = LinearRegression(fit_intercept=False).fit(X, y)
print(f'||OLS - sklearn|| = {np.linalg.norm(beta_hat - sk.coef_):.2e}')

# ─────────────────────────────────────────────
# 2. OLS의 표본분포 확인 — 정리 1.3
# ─────────────────────────────────────────────
n_trials = 5000
estimates = np.zeros((n_trials, p))
for t in range(n_trials):
    y_t = X @ beta_true + sigma_true * rng.standard_normal(n)
    estimates[t] = np.linalg.solve(X.T @ X, X.T @ y_t)

emp_mean = estimates.mean(axis=0)
emp_cov  = np.cov(estimates.T)
theo_cov = sigma_true ** 2 * np.linalg.inv(X.T @ X)

print(f'\n경험적 평균            : {emp_mean}')
print(f'이론값  E[β̂] = β      : {beta_true}')
print(f'\n공분산 최대 차이 ||emp - theo||_F = '
      f'{np.linalg.norm(emp_cov - theo_cov):.4e}')

# ─────────────────────────────────────────────
# 3. Gauss-Markov — 임의의 비편향 선형 추정량과 비교
# ─────────────────────────────────────────────
# 비편향 선형 추정량 한 예: 첫 p개 샘플만 사용 (X_sub가 정사각이라 unique)
def alt_estimator(y, X):
    X_sub, y_sub = X[:p], y[:p]   # 비편향이지만 비효율
    return np.linalg.solve(X_sub, y_sub)

ols_var, alt_var = [], []
for t in range(n_trials):
    y_t = X @ beta_true + sigma_true * rng.standard_normal(n)
    ols_var.append(np.linalg.solve(X.T @ X, X.T @ y_t))
    alt_var.append(alt_estimator(y_t, X))
ols_var = np.var(ols_var, axis=0)
alt_var = np.var(alt_var, axis=0)

print(f'\nOLS  분산: {ols_var}')
print(f'대안 분산: {alt_var}')
print(f'→ 모든 좌표에서 OLS 분산 ≤ 대안 분산 (Gauss-Markov)')
assert np.all(ols_var <= alt_var + 1e-6)

# ─────────────────────────────────────────────
# 4. σ² 비편향 추정 vs MLE (정리 1.5)
# ─────────────────────────────────────────────
resid = y - X @ beta_hat
sigma2_mle      = (resid ** 2).sum() / n
sigma2_unbiased = (resid ** 2).sum() / (n - p)
print(f'\nσ² 참값            : {sigma_true ** 2:.4f}')
print(f'σ²_MLE  (편향)     : {sigma2_mle:.4f}    (계수 (n-p)/n = {(n-p)/n:.3f})')
print(f'σ²_unbiased        : {sigma2_unbiased:.4f}')

# ─────────────────────────────────────────────
# 5. 실데이터 — California Housing
# ─────────────────────────────────────────────
data = fetch_california_housing()
X_real, y_real = data.data, data.target
X_real = np.hstack([np.ones((X_real.shape[0], 1)), X_real])  # bias 컬럼

beta_real = np.linalg.solve(X_real.T @ X_real, X_real.T @ y_real)
sk_real = LinearRegression(fit_intercept=False).fit(X_real, y_real)
print(f'\nCalifornia Housing 잔차 RMSE (NumPy):    '
      f'{np.sqrt(np.mean((y_real - X_real @ beta_real) ** 2)):.4f}')
print(f'                              (sklearn): '
      f'{np.sqrt(np.mean((y_real - sk_real.predict(X_real)) ** 2)):.4f}')
```

**출력 예시**:
```
OLS  : [ 1.018 -1.974  0.516 -0.027  3.001]
True : [ 1.   -2.    0.5   0.    3.  ]

||OLS - sklearn|| = 6.3e-15

경험적 평균            : [ 1.001 -2.000  0.499 -0.000  3.001]
이론값  E[β̂] = β      : [ 1.   -2.    0.5   0.    3.  ]

공분산 최대 차이 ||emp - theo||_F = 1.3e-03

OLS  분산: [0.00128 0.00134 0.00130 0.00135 0.00131]
대안 분산: [0.0612  0.0584  0.0721  0.0651  0.0589]
→ 모든 좌표에서 OLS 분산 ≤ 대안 분산 (Gauss-Markov)

σ² 참값            : 0.2500
σ²_MLE  (편향)     : 0.2434    (계수 (n-p)/n = 0.975)
σ²_unbiased        : 0.2496

California Housing 잔차 RMSE (NumPy):    0.7236
                              (sklearn): 0.7236
```

---

## 🔗 실전 활용

- **모델 진단의 출발점**: 잔차 $r = y - X\hat{\beta}$가 평균 0, 등분산, 비상관, 정규성 — 네 가정 위반은 잔차 플롯·QQ plot·Durbin-Watson으로 검사.
- **표준오차와 신뢰구간**: $\widehat{\text{SE}}(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2 [(X^\top X)^{-1}]_{jj}}$. 95% CI는 $\hat{\beta}_j \pm 1.96 \cdot \widehat{\text{SE}}$. **statsmodels.OLS가 자동 계산**.
- **다공선성(multicollinearity)**: $X^\top X$의 조건수가 크면 $(X^\top X)^{-1}$의 원소가 폭발 → $\hat{\beta}$의 분산 폭발. **VIF**(Variance Inflation Factor)와 **Ridge regression**(Ch1-04)으로 대응.
- **잔차 가정 위반 시 대체**: 잡음이 비등분산 → **GLS**(Generalized Least Squares), 잡음에 outlier 多 → **Huber Regression**, 비선형 → **Generalized Additive Model (GAM)**.

---

## ⚖️ 가정과 한계

| 가정 | 위반 시 | 대응 |
|------|--------|------|
| 선형성 $\mathbb{E}[y\|x] = x^\top \beta$ | 모델이 truth를 표현 불가 | 다항·spline·tree 기반 |
| $\epsilon \perp X$ | $\hat{\beta}$가 일치성 잃음 (편향) | **IV (Instrumental Variable)** |
| 등분산성 (homoskedasticity) | OLS는 여전히 비편향이지만 SE가 잘못됨 | **GLS** 또는 **HC robust SE** |
| 잡음 비상관 | OLS가 BLUE 잃음 (시계열에 흔함) | **GLS**, **Newey-West SE** |
| Gaussian 잡음 | MLE = OLS는 깨짐, BLUE는 유지 | **MLE under correct dist** 또는 **M-estimator** |
| $X$ full column rank | 해 비유일 → Pseudoinverse로 min-norm | **Ch1-03**, **Ridge** |

**주의**: "선형 회귀"의 "선형"은 **$\beta$에 대한 선형**이지 **$x$에 대한 선형**이 아니다. $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 \sin x$도 OLS의 영역 — feature engineering으로 비선형성을 흡수.

---

## 📌 핵심 정리

$$\boxed{y = X\beta + \epsilon,\ \epsilon \sim \mathcal{N}(0, \sigma^2 I) \implies \hat{\beta}_{\text{MLE}} = \arg\min_\beta \|y - X\beta\|^2 = (X^\top X)^{-1} X^\top y}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **MLE = OLS** | Gaussian noise 하에서 log-likelihood = $-\frac{1}{2\sigma^2} \mathrm{RSS}$ + 상수 |
| **Normal Equation** | $\nabla \mathrm{RSS} = 0 \iff X^\top X \beta = X^\top y$ — 잔차가 $X$의 열에 수직 |
| **OLS 분포** | $\hat{\beta} \sim \mathcal{N}(\beta, \sigma^2 (X^\top X)^{-1})$ — 비편향, 분산 명시적 |
| **Gauss-Markov** | 정규성 없이도 OLS = 모든 선형 비편향 추정량 중 최소분산 |
| **σ² 비편향 추정** | $\hat{\sigma}^2 = \mathrm{RSS}/(n - p)$ — MLE는 $1/n$로 편향 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $X = \mathbf{1}_n$ (모든 원소 1인 $n \times 1$ 벡터)일 때 $\hat{\beta}$가 표본평균 $\bar{y}$임을 정의로 직접 보여라.

<details>
<summary>힌트 및 해설</summary>

$X^\top X = \mathbf{1}^\top \mathbf{1} = n$, $X^\top y = \mathbf{1}^\top y = \sum y_i$. 따라서 $\hat{\beta} = (X^\top X)^{-1} X^\top y = \frac{1}{n}\sum y_i = \bar{y}$.

**의미**: 표본평균은 **상수 모델 $y \approx \beta$의 OLS 해**. 가장 단순한 회귀에서도 우리는 "MLE = 표본평균"이라는 익숙한 결과를 회복한다.

</details>

**문제 2** (심화): Gaussian 가정을 **Laplace** $\epsilon_i \sim \text{Laplace}(0, b)$로 바꾸면 MLE는 어떻게 되는가? 그 추정량을 무엇이라 부르는가?

<details>
<summary>힌트 및 해설</summary>

Laplace 밀도 $p(\epsilon) = \frac{1}{2b}\exp(-\|\epsilon\|/b)$. log-likelihood:

$$\ell(\beta) = -n\log(2b) - \frac{1}{b}\sum_i |y_i - x_i^\top \beta|$$

$\beta$ 부분은 $-\frac{1}{b}\sum_i |y_i - x_i^\top \beta|$ — **L1 절댓값 합 최소화**, 즉 **Least Absolute Deviations (LAD)** 또는 **Median Regression**. closed-form이 없고 LP로 푼다. **outlier에 robust**한 점이 OLS와의 차이. Quantile regression의 특수 사례 ($\tau = 0.5$).

</details>

**문제 3** (ML 연결): 신경망의 마지막 layer가 선형이고 손실이 MSE라면, 다른 weight들을 고정했을 때 마지막 layer의 weight가 OLS 해 $\hat{\beta} = (\Phi^\top \Phi)^{-1} \Phi^\top y$임을 설명하라 ($\Phi$는 마지막 hidden layer의 활성).

<details>
<summary>힌트 및 해설</summary>

마지막 hidden layer의 출력을 $\phi(x) \in \mathbb{R}^d$라 하면, NN의 예측은 $f(x) = w^\top \phi(x)$. 손실 $\sum_i (y_i - w^\top \phi(x_i))^2$를 $w$로 미분 = 0 → $\Phi^\top \Phi w = \Phi^\top y$. 즉 **NN의 마지막 layer = $\phi(x)$를 feature로 한 OLS**. 이것이 **Random Features** (Ch7 in Kernel Methods 레포), **Extreme Learning Machine**, **Neural Tangent Kernel** 모두의 출발점이다. 즉 "선형 회귀를 마스터하면 NN의 마지막 layer를 마스터한 셈"이다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [📚 README로 돌아가기](../README.md) | | [02. 기하학적 관점 — 수직투영 ▶](./02-geometric-projection.md) |

</div>
