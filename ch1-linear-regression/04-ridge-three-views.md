# 04. Ridge Regression의 3가지 해석

## 🎯 핵심 질문

- $\hat{\beta}_R = (X^\top X + \lambda I)^{-1} X^\top y$가 어떻게 (1) 정규화된 손실, (2) Gaussian prior MAP, (3) 제약 최적화의 **세 가지 동등한 표현**을 가지는가?
- 세 해석이 같은 식을 내려면 $\lambda$, prior 분산 $\tau^2$, 제약 반경 $c$가 어떻게 매핑되는가?
- SVD로 본 Ridge: 작은 특이값 방향을 어떻게 **shrinkage** 하는가? $\hat{\beta}_R = \sum \frac{\sigma_i}{\sigma_i^2 + \lambda}(u_i^\top y) v_i$의 의미는?
- $\lambda$를 cross-validation으로 어떻게 선택하는가? Generalized CV의 closed-form은?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Ridge regression은 (a) 다공선성 데이터에서 OLS의 분산을 직접 줄이는 가장 단순한 방법, (b) **확률적·기하학적·최적화적 세 시점**을 한 식이 동시에 갖는 보기 드문 사례, (c) 신경망 weight decay·SVM의 $\frac{1}{2}\|w\|^2$·Kernel Ridge·Gaussian Process posterior mean의 **공통 수학적 코어**. 즉 "$L_2$ penalty가 무엇을 하는가"의 완전한 그림이 여기 있다. 사용자 입장에서는 `Ridge(alpha=1.0)`이라는 한 줄이지만, 그 뒤에는 (i) $\lambda$가 **prior 강도**라는 Bayesian 해석, (ii) **L2 ball 위 사영**이라는 기하 해석, (iii) **SVD 방향별 차등 shrinkage**라는 최적화 해석이 동시에 작동한다. 이 세 해석을 잇는 능력 = 정규화의 본질을 이해한 것.

---

## 📐 수학적 선행 조건

- Normal Equation의 미분 유도 (Ch1-01)
- SVD와 pseudoinverse (Ch1-03)
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): MAP 추정, conjugate prior
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): Lagrangian, KKT, strong duality

---

## 📖 직관적 이해

### 해석 1 — 정규화: "큰 계수에 벌금"

$$\min_\beta \|y - X\beta\|^2 + \lambda \|\beta\|^2.$$

$\lambda > 0$이면 $\beta$가 클수록 손실이 커진다. **제약 없는 회귀에 norm penalty 추가**. 미분 = 0 → $X^\top X \beta + \lambda \beta = X^\top y$ → $(X^\top X + \lambda I)\beta = X^\top y$.

### 해석 2 — Bayesian: "prior가 0 근처에 몰린다고 가정"

$$\beta \sim \mathcal{N}(0, \tau^2 I), \qquad y \mid \beta \sim \mathcal{N}(X\beta, \sigma^2 I).$$

posterior $p(\beta \mid y) \propto p(y \mid \beta) p(\beta) \propto \exp\!\left(-\frac{1}{2\sigma^2}\|y - X\beta\|^2 - \frac{1}{2\tau^2}\|\beta\|^2\right)$. **MAP** = posterior mode = log posterior 최대화 = $\frac{1}{2\sigma^2}\|y - X\beta\|^2 + \frac{1}{2\tau^2}\|\beta\|^2$ 최소화. $\lambda := \sigma^2/\tau^2$로 놓으면 해석 1과 일치.

### 해석 3 — 제약: "norm을 c 이하로 묶어두고 최소제곱"

$$\min_\beta \|y - X\beta\|^2 \ \text{s.t.}\ \|\beta\|^2 \leq c^2.$$

라그랑지안 $\|y - X\beta\|^2 + \lambda(\|\beta\|^2 - c^2)$. KKT의 stationarity가 해석 1과 같은 식 — **$\lambda$는 라그랑주 승수**, $c$가 작을수록 $\lambda$ 큼 (둘은 일대일).

### SVD로 펼친 의미

$X = U\Sigma V^\top$, OLS는 $\hat{\beta}_{\text{OLS}} = \sum \frac{1}{\sigma_i}(u_i^\top y) v_i$. Ridge는

$$\hat{\beta}_R = \sum_{i=1}^p \frac{\sigma_i}{\sigma_i^2 + \lambda}(u_i^\top y) v_i.$$

shrinkage factor $f_i = \sigma_i^2/(\sigma_i^2 + \lambda) \in (0, 1)$가 OLS 계수에 곱해진다. **큰 $\sigma_i$**(데이터가 강하게 정보를 주는 방향)는 거의 1배, **작은 $\sigma_i$**(noise-prone 방향)는 강하게 축소.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Ridge 손실과 추정량

$\lambda > 0$이 주어지면

$$\hat{\beta}_R(\lambda) := \arg\min_\beta \mathcal{L}_R(\beta; \lambda), \qquad \mathcal{L}_R(\beta; \lambda) := \|y - X\beta\|^2 + \lambda \|\beta\|^2.$$

Closed-form: $\hat{\beta}_R(\lambda) = (X^\top X + \lambda I)^{-1} X^\top y$.

### 정의 4.2 — Gaussian Prior MAP

데이터 모델 $y \mid \beta \sim \mathcal{N}(X\beta, \sigma^2 I)$, prior $\beta \sim \mathcal{N}(0, \tau^2 I)$. MAP 추정량은 $\arg\max_\beta p(\beta \mid y)$.

### 정의 4.3 — Constrained Form

반경 $c > 0$이 주어지면

$$\hat{\beta}_C(c) := \arg\min_\beta \|y - X\beta\|^2 \ \text{s.t.}\ \|\beta\|^2 \leq c^2.$$

---

## 🔬 정리와 증명

### 정리 4.1 — Ridge 정규방정식

**명제**: $\lambda > 0$이면 $\mathcal{L}_R$의 유일한 최소점은 $\hat{\beta}_R = (X^\top X + \lambda I)^{-1} X^\top y$.

**증명**: $\nabla_\beta \mathcal{L}_R = -2 X^\top (y - X\beta) + 2\lambda \beta = 0 \Rightarrow (X^\top X + \lambda I)\beta = X^\top y$. $X^\top X + \lambda I$는 임의의 $X$에 대해 PD ($v^\top (X^\top X + \lambda I) v = \|Xv\|^2 + \lambda \|v\|^2 \geq \lambda \|v\|^2 > 0$ for $v \neq 0$) → 가역. 2계 조건 $\nabla^2 \mathcal{L}_R = 2(X^\top X + \lambda I) \succ 0$ → 강볼록 → 유일 최소. $\square$

> 💡 **중요**: Ridge는 $X^\top X$가 특이여도 항상 unique solution. 이것이 OLS 대비 가장 큰 실무적 장점.

### 정리 4.2 — MAP = Ridge

**명제**: 정의 4.2의 MAP 추정량은 $\hat{\beta}_R(\lambda)$ with $\lambda = \sigma^2/\tau^2$.

**증명**: posterior log-density:

$$\log p(\beta \mid y) = -\frac{1}{2\sigma^2}\|y - X\beta\|^2 - \frac{1}{2\tau^2}\|\beta\|^2 + \text{const}.$$

이를 최대화 = $\frac{1}{2\sigma^2}\|y - X\beta\|^2 + \frac{1}{2\tau^2}\|\beta\|^2$ 최소화 = $2\sigma^2$ 곱해도 같음 = $\|y - X\beta\|^2 + \frac{\sigma^2}{\tau^2}\|\beta\|^2$ 최소화. 따라서 $\lambda = \sigma^2/\tau^2$. $\square$

> 📌 **Bayesian 해석**: $\tau^2 \to \infty$ (uninformative prior) → $\lambda \to 0$ → OLS 회복. $\tau^2 \to 0$ (delta prior at 0) → $\lambda \to \infty$ → $\hat{\beta} \to 0$. **prior 강도가 정규화 강도**.

### 정리 4.3 — Constrained = Ridge (라그랑지안 동치)

**명제**: 정의 4.3에 대해, 임의의 $c > 0$에 대해 어떤 $\lambda \geq 0$이 존재해 $\hat{\beta}_C(c) = \hat{\beta}_R(\lambda)$. 거꾸로 임의의 $\lambda > 0$에 대해 어떤 $c > 0$이 존재해 같은 등치.

**증명**: 라그랑지안 $L(\beta, \lambda) = \|y - X\beta\|^2 + \lambda(\|\beta\|^2 - c^2)$. KKT 조건:

1. **Stationarity**: $\nabla_\beta L = -2X^\top(y - X\beta) + 2\lambda \beta = 0$ → 정리 4.1.
2. **Primal feasibility**: $\|\beta\|^2 \leq c^2$.
3. **Dual feasibility**: $\lambda \geq 0$.
4. **Complementary slackness**: $\lambda(\|\beta\|^2 - c^2) = 0$.

원문제는 strong duality 만족 (목적함수와 제약 모두 볼록, Slater $\beta = 0$이 strict feasible). 따라서 KKT가 충분조건. $c$가 OLS 해의 norm $\|\hat{\beta}_{\text{OLS}}\|$ 이상이면 $\lambda = 0$ (제약 비활성), 그 외 $\lambda > 0$이고 $\|\beta_R\| = c$ (active constraint). 일대일 대응. $\square$

> 💡 **기하적 그림**: L2 제약 영역은 $\mathbb{R}^p$의 **공(球)**. OLS 등고선(타원체)이 공과 만나는 첫 점이 $\hat{\beta}_R$. $c$를 줄이면 공이 작아지고 해가 0으로 수렴.

### 정리 4.4 — SVD를 통한 Ridge의 펼침

**명제**: $X = U \Sigma V^\top$ (thin SVD, $\sigma_1 \geq \cdots \geq \sigma_r > 0$). 그러면

$$\hat{\beta}_R(\lambda) = \sum_{i=1}^r \frac{\sigma_i}{\sigma_i^2 + \lambda}(u_i^\top y) \, v_i.$$

또 fitted value $\hat{y}_R = X\hat{\beta}_R$는

$$\hat{y}_R = \sum_{i=1}^r \frac{\sigma_i^2}{\sigma_i^2 + \lambda}(u_i^\top y) \, u_i.$$

**증명**: $X^\top X + \lambda I = V(\Sigma^\top \Sigma + \lambda I)V^\top + \lambda V_\perp V_\perp^\top$ (V_⊥는 $V$의 영공간 부분). 첫 항만 OLS와 곂쳐서

$$(X^\top X + \lambda I)^{-1} X^\top = V\,\text{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right) U^\top.$$

이를 $y$에 곱해 $\hat{\beta}_R$. $X\hat{\beta}_R = U\Sigma V^\top V \,\text{diag}(\sigma_i/(\sigma_i^2 + \lambda)) U^\top y = U \,\text{diag}(\sigma_i^2/(\sigma_i^2 + \lambda)) U^\top y$. $\square$

> 📌 **shrinkage factor 해석**: $f_i(\lambda) = \sigma_i^2/(\sigma_i^2 + \lambda)$. $\sigma_i \gg \sqrt{\lambda}$ → $f_i \approx 1$ (정보 풍부 방향은 보존). $\sigma_i \ll \sqrt{\lambda}$ → $f_i \approx 0$ (잡음 방향은 제거). **PCA-스러운 효과를 자동으로**.

### 정리 4.5 — Ridge의 Bias-Variance Trade-off

**명제**: 모델 1.1 하에서

$$\mathbb{E}[\hat{\beta}_R] = (X^\top X + \lambda I)^{-1} X^\top X \beta = \beta - \lambda (X^\top X + \lambda I)^{-1} \beta,$$

$$\text{Var}(\hat{\beta}_R) = \sigma^2 (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1}.$$

따라서 $\lambda > 0$이면 **편향 발생**하지만 **분산 감소**.

**증명**: $\hat{\beta}_R = (X^\top X + \lambda I)^{-1} X^\top (X\beta + \epsilon)$ → 평균과 분산은 affine 변환의 표준 공식. SVD로 펼치면 좌표별로 $\text{Bias}_i = \frac{-\lambda}{\sigma_i^2 + \lambda} v_i^\top \beta$, $\text{Var}_i = \frac{\sigma^2 \sigma_i^2}{(\sigma_i^2 + \lambda)^2}$. $\square$

> 💡 **Optimal λ (oracle)**: 단일 좌표 $\text{MSE}_i = \text{Bias}_i^2 + \text{Var}_i$의 최소를 미분하면 $\lambda^* = \sigma^2/(v_i^\top \beta)^2$. 좌표마다 다른 최적이지만, **글로벌 $\lambda$도 양의 값에서 OLS보다 낮은 MSE**를 줄 수 있다 (편향-분산 거래의 본질).

### 정리 4.6 — Generalized Cross-Validation (GCV)

**명제**: $\lambda$를 선택하기 위한 GCV 스코어:

$$\text{GCV}(\lambda) = \frac{1}{n} \sum_{i=1}^n \left(\frac{y_i - \hat{y}_i(\lambda)}{1 - \text{tr}(H_R(\lambda))/n}\right)^2,$$

여기서 $H_R(\lambda) = X(X^\top X + \lambda I)^{-1} X^\top$. GCV는 **leave-one-out CV의 closed-form 근사**.

**증명 스케치**: LOO error는 $\sum (y_i - \hat{y}_i^{(-i)})^2$. Sherman-Morrison으로 $\hat{y}_i^{(-i)} = (\hat{y}_i - h_{ii} y_i)/(1 - h_{ii})$. LOO error를 $\sum (y_i - \hat{y}_i)^2/(1 - h_{ii})^2$로 펼친 후 $h_{ii}$를 평균 $\text{tr}(H)/n$로 대체한 것이 GCV. $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. 세 해석이 같은 해를 내는지
# ─────────────────────────────────────────────
n, p = 100, 10
X = rng.standard_normal((n, p))
beta_true = rng.standard_normal(p)
sigma = 1.0
y = X @ beta_true + sigma * rng.standard_normal(n)

# (1) 정규화: closed-form
lam = 2.0
beta_reg = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)

# (2) MAP: posterior mode (사실상 같은 식)
tau2 = sigma**2 / lam   # λ = σ²/τ² 매핑
A = X.T @ X / sigma**2 + np.eye(p) / tau2
b = X.T @ y / sigma**2
beta_map = np.linalg.solve(A, b)

# (3) 제약: cvxpy로 해결
import cvxpy as cp
beta_var = cp.Variable(p)
norm_constraint = np.linalg.norm(beta_reg)   # 정확한 c를 안다면
prob = cp.Problem(cp.Minimize(cp.sum_squares(y - X @ beta_var)),
                  [cp.norm(beta_var, 2) <= norm_constraint])
prob.solve()
beta_con = beta_var.value

# sklearn 비교 — Ridge(alpha)에서 alpha = λ
sk = Ridge(alpha=lam, fit_intercept=False, solver='cholesky').fit(X, y)

print(f'정규화 (closed-form): {beta_reg[:3]}')
print(f'MAP                 : {beta_map[:3]}')
print(f'제약 (cvxpy)        : {beta_con[:3]}')
print(f'sklearn Ridge       : {sk.coef_[:3]}')
print(f'\n||reg - map||  = {np.linalg.norm(beta_reg - beta_map):.2e}')
print(f'||reg - con||  = {np.linalg.norm(beta_reg - beta_con):.2e}')
print(f'||reg - sklearn|| = {np.linalg.norm(beta_reg - sk.coef_):.2e}')

# ─────────────────────────────────────────────
# 2. SVD로 shrinkage factor 시각화 (정리 4.4)
# ─────────────────────────────────────────────
U, s, Vt = np.linalg.svd(X, full_matrices=False)
print(f'\n특이값들: {s.round(3)}')
for lam_test in [0.01, 1.0, 100.0]:
    f = s**2 / (s**2 + lam_test)
    print(f'λ = {lam_test:>6}:  shrinkage = {f.round(3)}')

# ─────────────────────────────────────────────
# 3. Bias-Variance trade-off (정리 4.5)
# ─────────────────────────────────────────────
lambdas = np.logspace(-2, 3, 30)
n_trials = 1000
biases, variances, mses = [], [], []
for lam in lambdas:
    estimates = np.zeros((n_trials, p))
    for t in range(n_trials):
        eps = sigma * rng.standard_normal(n)
        y_t = X @ beta_true + eps
        estimates[t] = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y_t)
    bias = np.mean(estimates, axis=0) - beta_true
    var  = np.var(estimates, axis=0)
    biases.append(np.sum(bias**2))
    variances.append(np.sum(var))
    mses.append(np.sum(bias**2 + var))

opt_idx = np.argmin(mses)
print(f'\n최적 λ ≈ {lambdas[opt_idx]:.4f}, MSE = {mses[opt_idx]:.4f}')
print(f'OLS  (λ→0): MSE ≈ {mses[0]:.4f}')

# ─────────────────────────────────────────────
# 4. Generalized CV (정리 4.6)
# ─────────────────────────────────────────────
def gcv(X, y, lam):
    H = X @ np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T)
    yhat = H @ y
    return np.mean(((y - yhat) / (1 - np.trace(H)/len(y)))**2)

gcv_scores = [gcv(X, y, l) for l in lambdas]
opt_gcv = lambdas[np.argmin(gcv_scores)]
print(f'GCV-선택 λ = {opt_gcv:.4f}')
```

**출력 예시**:
```
정규화 (closed-form): [-0.215  0.418 -0.123]
MAP                 : [-0.215  0.418 -0.123]
제약 (cvxpy)        : [-0.215  0.418 -0.123]
sklearn Ridge       : [-0.215  0.418 -0.123]

||reg - map||  = 1.39e-15
||reg - con||  = 4.21e-08   (cvxpy 수치 정밀도)
||reg - sklearn|| = 0.00e+00

특이값들: [12.21 11.97 11.34 11.01 10.42  9.86  9.43  8.95  8.32  7.51]
λ = 0.01:  shrinkage = [1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000]
λ = 1.00:  shrinkage = [0.993 0.993 0.992 0.992 0.991 0.990 0.989 0.988 0.986 0.983]
λ = 100.0: shrinkage = [0.598 0.589 0.563 0.548 0.520 0.493 0.471 0.445 0.409 0.360]

최적 λ ≈ 5.6234, MSE = 0.0641
OLS  (λ→0): MSE ≈ 0.1041
GCV-선택 λ = 4.8329
```

---

## 🔗 실전 활용

- **표준 정규화 기법**: 거의 모든 선형 모델 라이브러리에 기본 옵션. sklearn `Ridge(alpha=...)`.
- **표준화의 중요성**: $\lambda \|\beta\|^2$는 **모든 좌표를 동등하게 penalize**. feature가 다른 단위면(예: 키 cm vs 체중 kg), 단위가 큰 feature가 부당하게 보호됨 → **반드시 standardize** ($X_j \leftarrow (X_j - \bar{X}_j)/s_j$).
- **Bias 항 별도 처리**: 절편 $\beta_0$는 보통 penalty에서 제외. sklearn이 `fit_intercept=True`일 때 자동.
- **High-dim 회귀**: $n < p$여도 Ridge는 항상 unique. genomics·image regression의 표준.
- **Kernel Ridge로 일반화**: $X^\top X \to K$ (Gram matrix), pseudoinverse 대신 $(K + \lambda I)^{-1}$ — 비선형 회귀의 표준 (Kernel Methods 레포 Ch5).
- **Gaussian Process posterior mean과 동치**: $m_*(x) = k(x, X)(K + \sigma_n^2 I)^{-1} y$ — Ridge의 $\lambda = \sigma_n^2$.

---

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 |
|------------|------|
| L2 penalty | 모든 $\beta_j$를 비례적으로 줄임 — sparsity 없음 (이건 Lasso의 영역) |
| Gaussian prior | "0 근처가 더 그럴듯하다"는 가정이 도메인에 맞아야 |
| 글로벌 $\lambda$ | 좌표별로 다른 $\lambda_j$가 더 좋을 수 있음 → **Adaptive Ridge** |
| 표준화 의존 | feature scale에 민감 |

**주의**: $\lambda$를 train data로 선택하면 leak. CV/GCV 또는 hold-out validation set 필수.

---

## 📌 핵심 정리

$$\boxed{\hat{\beta}_R = (X^\top X + \lambda I)^{-1} X^\top y = \arg\min \|y - X\beta\|^2 + \lambda \|\beta\|^2 = \text{MAP}\bigl(\beta \sim \mathcal{N}(0, \tau^2 I)\bigr)\bigr|_{\lambda = \sigma^2/\tau^2}}$$

| 해석 | 형태 | 매핑 |
|------|------|------|
| 정규화 | $\min \|y - X\beta\|^2 + \lambda \|\beta\|^2$ | $\lambda$ 직접 |
| MAP (Bayesian) | $\beta \sim \mathcal{N}(0, \tau^2 I)$, $y\|\beta \sim \mathcal{N}(X\beta, \sigma^2 I)$ | $\lambda = \sigma^2/\tau^2$ |
| 제약 | $\min \|y - X\beta\|^2$ s.t. $\|\beta\| \leq c$ | $c \leftrightarrow \lambda$ (KKT) |
| SVD | $\hat{\beta}_R = \sum \frac{\sigma_i}{\sigma_i^2 + \lambda} u_i^\top y\, v_i$ | shrinkage factor $\sigma_i^2/(\sigma_i^2+\lambda)$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\lambda \to \infty$와 $\lambda \to 0^+$에서 $\hat{\beta}_R$의 극한을 각각 구하라.

<details>
<summary>힌트 및 해설</summary>

$\lambda \to 0^+$: 정리 3.3에 의해 $\hat{\beta}_R \to X^+ y$ (pseudoinverse 해, full rank이면 OLS).

$\lambda \to \infty$: $(X^\top X + \lambda I)^{-1} \approx \lambda^{-1} I$ → $\hat{\beta}_R \approx \lambda^{-1} X^\top y \to 0$. 모든 계수가 0으로 — **완전 정규화**.

</details>

**문제 2** (심화): 정리 4.4의 SVD 표현을 사용해 **단순회귀**(p = 1)에서 $\hat{\beta}_R$의 closed-form을 구하라. 그리고 OLS 대비 어떻게 축소되는지 보여라.

<details>
<summary>힌트 및 해설</summary>

$X \in \mathbb{R}^{n \times 1}$, $\sigma_1 = \|X\|$. $u_1 = X/\|X\|$. $v_1 = 1$ (스칼라).

OLS: $\hat{\beta}_{\text{OLS}} = \frac{X^\top y}{\|X\|^2}$.

Ridge: $\hat{\beta}_R = \frac{\|X\|}{\|X\|^2 + \lambda} \cdot \frac{X^\top y}{\|X\|} = \frac{X^\top y}{\|X\|^2 + \lambda} = \hat{\beta}_{\text{OLS}} \cdot \frac{\|X\|^2}{\|X\|^2 + \lambda}$.

→ 항상 OLS의 **양의 배수 (< 1)** 만큼 축소. $\lambda$가 데이터 분산 $\|X\|^2$에 비해 클 때 강하게 축소.

</details>

**문제 3** (ML 연결): NN의 **weight decay** $L = \mathcal{L}_{\text{data}} + \frac{\lambda}{2}\sum \|w_l\|^2$가 본 문서의 어떤 해석에 해당하는지 설명하라. NN에서 "Bayesian prior" 해석은 무엇인가?

<details>
<summary>힌트 및 해설</summary>

**Weight decay = Ridge의 NN 일반화** — 모든 weight에 L2 penalty.

해석 1 (정규화): 명백.

해석 2 (MAP): 각 weight $w_l \sim \mathcal{N}(0, \tau^2 I)$ prior 하의 MAP. 이것이 **Bayesian Neural Network**의 가장 단순한 prior, **Variational Inference** + **mean-field Gaussian** 의 출발점.

해석 3 (제약): NN의 weight를 ball 안으로 제약 = Lipschitz 제어 = generalization bound.

따라서 **"NN에 weight decay 0.01을 주는 것" = "각 weight가 평균 0, 분산 1/0.01의 Gaussian이라는 prior 가정 + MAP 추정"**. PyTorch optimizer의 `weight_decay` 파라미터가 사실 Bayesian 추론.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Pseudoinverse](./03-pseudoinverse.md) | [📚 README](../README.md) | [05. Lasso와 Sparsity ▶](./05-lasso-sparsity.md) |

</div>
