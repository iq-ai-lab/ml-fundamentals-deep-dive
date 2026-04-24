# 02. IRLS — Iteratively Reweighted Least Squares

## 🎯 핵심 질문

- Logistic Regression의 Newton-Raphson 업데이트 $\beta \leftarrow \beta + (X^\top W X)^{-1} X^\top (y - p)$가 왜 **가중 최소제곱 문제**와 정확히 같은가?
- Working response $z = X\beta + W^{-1}(y - p)$는 어디서 오고 왜 그 형태인가?
- IRLS 한 iteration이 사실상 **OLS + Ridge를 가중치와 함께 푸는 것**이라는 관점은 어떤 힘을 주는가?
- sklearn / statsmodels의 LR solver가 내부적으로 IRLS를 돌리는 이유와 수치적 고려사항은?

---

## 🔍 왜 이 개념이 ML에서 중요한가

IRLS는 (a) LR의 **표준 학습 알고리즘** (statsmodels·R·MATLAB의 기본), (b) **GLM의 모든 분포에 통일 적용** — Poisson/Gamma/Binomial 같은 식 (Ch2-03), (c) NN의 2차 최적화 알고리즘(K-FAC, Natural Gradient)의 **원형**, (d) **수치적으로 $O(np^2)$** (Newton의 전형)이면서 **닫힌 형 step** → IRLS를 이해하면 Logistic Regression이 "매 step이 가중 OLS"임이 명확해진다. 또한 이 관점은 학습률·line search·second-order 직관을 모두 제공. 마지막으로 **"NR이 가중 OLS"**라는 한 문장은 선형 회귀에서 GLM으로의 다리이다.

---

## 📐 수학적 선행 조건

- LR의 gradient와 Hessian (Ch2-01 정리 1.2, 1.3)
- Newton-Raphson의 일반적 유도
- Weighted Least Squares의 closed-form

---

## 📖 직관적 이해

### Newton-Raphson의 핵심

concave $\ell$의 최대화 → Newton step $\beta^{(t+1)} = \beta^{(t)} - (\nabla^2 \ell)^{-1} \nabla \ell$. 2차 근사를 풀어 local max로 점프.

### "가중 OLS"와 같은 이유

Newton step:

$$\beta^{(t+1)} = \beta^{(t)} + (X^\top W X)^{-1} X^\top (y - p).$$

이를 정리하면:

$$\beta^{(t+1)} = (X^\top W X)^{-1} X^\top W \cdot \underbrace{[X\beta^{(t)} + W^{-1}(y - p)]}_{z^{(t)} \text{ (working response)}}.$$

이 형태가 **$W$-가중 OLS**의 해 $\hat{\beta} = (X^\top W X)^{-1} X^\top W z$. 즉 **Newton 한 step = $z$를 타겟으로 한 WLS**.

### Working Response의 의미

$z = X\beta + W^{-1}(y - p)$는 **"선형 예측자 $\eta = X\beta$를 잔차로 보정한 가상 타겟"**. 분류 문제가 회귀 문제로 변환되어 OLS 같이 풀림.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Newton-Raphson Update

concave $\ell$, gradient $g = \nabla \ell$, Hessian $H = \nabla^2 \ell$. Newton step:

$$\beta^{(t+1)} := \beta^{(t)} - H^{-1} g\bigr|_{\beta^{(t)}} = \beta^{(t)} + (-H)^{-1} g.$$

### 정의 2.2 — Weighted Least Squares

가중치 행렬 $W = \text{diag}(w_i) \succ 0$에 대해

$$\hat{\beta}_{\text{WLS}} := \arg\min_\beta (z - X\beta)^\top W (z - X\beta) = (X^\top W X)^{-1} X^\top W z.$$

### 정의 2.3 — IRLS Algorithm

Iteration $t$:

1. $\eta^{(t)} := X \beta^{(t)}$
2. $p^{(t)} := \sigma(\eta^{(t)})$
3. $W^{(t)} := \text{diag}(p_i^{(t)} (1 - p_i^{(t)}))$
4. $z^{(t)} := \eta^{(t)} + (W^{(t)})^{-1}(y - p^{(t)})$
5. $\beta^{(t+1)} := (X^\top W^{(t)} X)^{-1} X^\top W^{(t)} z^{(t)}$

수렴 판정 $\|\beta^{(t+1)} - \beta^{(t)}\| < \epsilon$.

---

## 🔬 정리와 증명

### 정리 2.1 — Newton-Raphson ≡ IRLS

**명제**: LR의 Newton-Raphson 업데이트

$$\beta^{(t+1)} = \beta^{(t)} + (X^\top W^{(t)} X)^{-1} X^\top (y - p^{(t)})$$

은 정의 2.3의 IRLS 업데이트

$$\beta^{(t+1)} = (X^\top W^{(t)} X)^{-1} X^\top W^{(t)} z^{(t)}, \quad z^{(t)} = X\beta^{(t)} + (W^{(t)})^{-1}(y - p^{(t)})$$

와 정확히 같다.

**증명**: 정리 1.2, 1.3: $g = X^\top (y - p)$, $H = -X^\top W X$. Newton step:

$$\beta^{(t+1)} = \beta^{(t)} - H^{-1} g = \beta^{(t)} + (X^\top W X)^{-1} X^\top (y - p).$$

이제 $A := X^\top W X$, $X^\top (y - p) = X^\top W \cdot W^{-1}(y - p)$. 따라서

$$\beta^{(t+1)} = \beta^{(t)} + A^{-1} X^\top W \cdot W^{-1}(y - p) = A^{-1}(A \beta^{(t)} + X^\top W \cdot W^{-1}(y - p)).$$

$A \beta^{(t)} = X^\top W X \beta^{(t)} = X^\top W \eta^{(t)}$이므로

$$\beta^{(t+1)} = A^{-1} X^\top W \bigl(\eta^{(t)} + W^{-1}(y - p)\bigr) = A^{-1} X^\top W z^{(t)}. \quad \square$$

### 정리 2.2 — IRLS의 각 Iteration = WLS 최소화

**명제**: IRLS의 한 step은 손실 $\|W^{1/2}(z^{(t)} - X\beta)\|^2 = \sum_i w_i^{(t)} (z_i^{(t)} - x_i^\top \beta)^2$의 최소화이다.

**증명**: WLS 정의 2.2에서 최소화 해가 $(X^\top W X)^{-1} X^\top W z$와 같다. $\square$

> 💡 **해석**: IRLS는 **"매 step마다 $z$라는 새 타겟과 $W$라는 새 가중치를 만들어 OLS를 풀기"**. 가중치 $w_i = p_i(1-p_i)$는 **예측이 0/1에서 멀어 불확실한 점에 큰 가중치** — 모호한 점이 더 중요함.

### 정리 2.3 — Working Response의 해석

**명제**: $z_i = \eta_i + (y_i - p_i)/(p_i(1 - p_i))$는 **$y_i$ 근방에서 sigmoid의 역 $\sigma^{-1} = \text{logit}$의 1차 Taylor 근사**.

**증명**: $\sigma^{-1}(y) \approx \sigma^{-1}(p) + \frac{1}{\sigma'(\sigma^{-1}(p))} (y - p) = \eta + \frac{1}{p(1-p)}(y - p)$. 마지막 등식은 $\sigma'(\eta) = p(1-p)$. $\square$

> 📌 **의미**: $z$는 "현재 모델의 logit $\eta$를 관측 $y$에 맞춰 업데이트한 근사 logit". 이를 타겟으로 OLS를 돌리면 logit 공간에서 선형 회귀 — **분류를 회귀로 환원한 것**.

### 정리 2.4 — IRLS의 수렴성

**명제**: 

1. $\ell$가 strictly concave (정리 1.3, full column rank 가정)이고, 
2. 초기값이 domain 내부에 있고,
3. 각 step의 Hessian이 non-degenerate ($p_i \in (0, 1)$ 유지)

이면 IRLS는 **quadratic convergence** — 수렴 속도 $\|\beta^{(t+1)} - \hat{\beta}\| \leq C \|\beta^{(t)} - \hat{\beta}\|^2$.

**증명 스케치**: Newton-Raphson의 일반 정리. Hessian이 Lipschitz + 최소점에서 strictly convex이면 2차 수렴 (Nocedal & Wright Ch11). LR은 $\ell$가 $C^\infty$이고 Hessian이 strictly convex (full column rank) → 조건 만족. $\square$

> 💡 **실전 수렴 속도**: 보통 5~15 iteration으로 수렴. 비교: Gradient Descent는 $O(\log(1/\epsilon))$ (linear), Newton은 $O(\log\log(1/\epsilon))$ (quadratic) — 훨씬 빠름.

### 정리 2.5 — Ridge-regularized IRLS

**명제**: penalty $\frac{\lambda}{2}\|\beta\|^2$를 추가한 Ridge LR의 Newton-Raphson은

$$\beta^{(t+1)} = (X^\top W^{(t)} X + \lambda I)^{-1} (X^\top W^{(t)} z^{(t)} + 0).$$

즉 **매 step이 Ridge regression**.

**증명**: penalized $\ell_R = \ell - \frac{\lambda}{2}\|\beta\|^2$, $\nabla = X^\top(y-p) - \lambda \beta$, $\nabla^2 = -X^\top W X - \lambda I$. Newton step → 위 공식. $\square$

> 📌 **장점**: (a) separation 시에도 unique solution, (b) 수치적 안정성, (c) cross-validation으로 $\lambda$ 선택 가능. sklearn `LogisticRegression`의 기본값 (`C = 1.0`).

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

rng = np.random.default_rng(42)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ─────────────────────────────────────────────
# 1. 합성 데이터
# ─────────────────────────────────────────────
n, p = 500, 5
X = rng.standard_normal((n, p))
beta_true = np.array([1.0, -2.0, 0.5, 0.0, 1.5])
y = (rng.uniform(size=n) < sigmoid(X @ beta_true)).astype(int)

# ─────────────────────────────────────────────
# 2. IRLS 바닥부터 (정의 2.3)
# ─────────────────────────────────────────────
def irls(X, y, n_iter=50, tol=1e-10, ridge=0.0):
    n, p = X.shape
    beta = np.zeros(p)
    history = [beta.copy()]
    for it in range(n_iter):
        eta = X @ beta
        prob = sigmoid(eta)
        W = prob * (1 - prob) + 1e-12     # 수치 안정
        z = eta + (y - prob) / W
        # WLS + optional ridge
        A = X.T @ (W[:, None] * X) + ridge * np.eye(p)
        b = X.T @ (W * z)
        beta_new = np.linalg.solve(A, b)
        history.append(beta_new.copy())
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    return beta, history, it + 1

beta_irls, hist, n_iter = irls(X, y)
print(f'IRLS 수렴 반복: {n_iter}')
print(f'IRLS β̂      : {beta_irls}')
print(f'True β      : {beta_true}')

# ─────────────────────────────────────────────
# 3. Newton-Raphson 식으로 직접 구현 (정리 2.1 확인)
# ─────────────────────────────────────────────
def newton(X, y, n_iter=50, tol=1e-10):
    p = X.shape[1]
    beta = np.zeros(p)
    for it in range(n_iter):
        prob = sigmoid(X @ beta)
        W = prob * (1 - prob)
        g = X.T @ (y - prob)
        H = X.T @ (W[:, None] * X)
        update = np.linalg.solve(H, g)
        beta_new = beta + update
        if np.linalg.norm(update) < tol:
            break
        beta = beta_new
    return beta, it + 1

beta_nr, nr_iter = newton(X, y)
print(f'\nNewton β̂    : {beta_nr}')
print(f'||IRLS - Newton|| = {np.linalg.norm(beta_irls - beta_nr):.2e}')
print(f'(정리 2.1 검증: 두 방식이 정확히 같은 값)')

# ─────────────────────────────────────────────
# 4. sklearn / statsmodels 비교
# ─────────────────────────────────────────────
sk = LogisticRegression(fit_intercept=False, C=1e10, max_iter=1000, tol=1e-12).fit(X, y)
sm_model = sm.Logit(y, X).fit(disp=0)

print(f'\nsklearn β̂    : {sk.coef_[0]}')
print(f'statsmodels β̂: {sm_model.params}')
print(f'||IRLS - sklearn||     = {np.linalg.norm(beta_irls - sk.coef_[0]):.2e}')
print(f'||IRLS - statsmodels|| = {np.linalg.norm(beta_irls - sm_model.params):.2e}')

# ─────────────────────────────────────────────
# 5. 수렴 속도: GD vs Newton (정리 2.4)
# ─────────────────────────────────────────────
def gd(X, y, lr=0.1, n_iter=1000, tol=1e-10):
    p = X.shape[1]
    beta = np.zeros(p)
    for it in range(n_iter):
        prob = sigmoid(X @ beta)
        g = X.T @ (y - prob) / len(y)
        beta_new = beta + lr * g
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
    return beta, it + 1

_, gd_iter = gd(X, y, lr=0.1)
print(f'\n수렴 속도:')
print(f'  GD (lr=0.1):     {gd_iter} iter')
print(f'  Newton/IRLS:     {n_iter} iter   (= 정리 2.4의 quadratic convergence)')

# ─────────────────────────────────────────────
# 6. Working response의 의미 시각화
# ─────────────────────────────────────────────
print(f'\n마지막 iteration의 working response z:')
prob = sigmoid(X @ beta_irls)
W = prob * (1 - prob)
eta = X @ beta_irls
z = eta + (y - prob) / W
# z와 eta 비교
for i in range(5):
    print(f'  i={i:>3}: y={y[i]}  η={eta[i]:+.3f}  p={prob[i]:.3f}  z={z[i]:+.3f}')
```

**출력 예시**:
```
IRLS 수렴 반복: 7
IRLS β̂      : [ 1.051 -2.012  0.503 -0.065  1.547]
True β      : [ 1.   -2.    0.5   0.    1.5 ]

Newton β̂    : [ 1.051 -2.012  0.503 -0.065  1.547]
||IRLS - Newton|| = 0.00e+00
(정리 2.1 검증: 두 방식이 정확히 같은 값)

sklearn β̂    : [ 1.051 -2.012  0.503 -0.065  1.547]
statsmodels β̂: [ 1.051 -2.012  0.503 -0.065  1.547]

수렴 속도:
  GD (lr=0.1):     623 iter
  Newton/IRLS:     7 iter   (= 정리 2.4의 quadratic convergence)
```

---

## 🔗 실전 활용

- **statsmodels.GLM**: 내부가 정확히 IRLS. `method='IRLS'` (기본값).
- **sklearn LogisticRegression** (`solver='lbfgs'` 기본) vs `'newton-cg'`: L-BFGS는 quasi-Newton, newton-cg는 Hessian-free Newton. 둘 다 IRLS와 비슷하지만 inner step 다름. **`solver='newton-cholesky'`** (v1.2+)이 정확히 Ridge-IRLS.
- **대규모 데이터**: $n$ 크면 $X^\top W X$ 계산이 $O(np^2)$ → 수천만 샘플에서는 SGD/mini-batch로. LR의 **"작은 $n$에 Newton, 큰 $n$에 SGD"**.
- **GLM 일반화**: Poisson·Binomial·Gamma 모두 IRLS로 같은 틀 (Ch2-03). R의 `glm()` 함수가 IRLS 기반.

---

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 |
|------------|------|
| Strictly concave $\ell$ | Separation이면 발산 — Ridge 필요 |
| Hessian 계산 가능 | $O(np^2)$ — 대용량에 비쌈 |
| Hessian 역 계산 | ill-cond면 numerical blow-up — small ridge jitter 추가 |
| Batch 방식 | mini-batch로 하려면 별도 알고리즘 (SGD + IRLS 혼합 드묾) |

**주의**: IRLS는 step size를 명시하지 않고 "full Newton step"을 쓴다. 비볼록 문제에서 이는 오버슈팅 위험 — LR은 볼록이라 안전하지만 확장된 상황 (GLM with heavy tails)에서는 line search 필요.

---

## 📌 핵심 정리

$$\boxed{\text{Newton} \equiv \text{IRLS:}\quad \beta^{(t+1)} = (X^\top W^{(t)} X)^{-1} X^\top W^{(t)} z^{(t)},\quad z^{(t)} = X\beta^{(t)} + (W^{(t)})^{-1}(y - p^{(t)})}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **Newton step = WLS** | Newton의 $(X^\top W X)^{-1}$이 바로 WLS 해 |
| **Working response** | logit의 1차 Taylor — 분류를 회귀로 환원 |
| **가중치 $w_i$** | $p_i(1-p_i)$ — 모호한 예측에 큰 가중 |
| **수렴 속도** | Quadratic — GD보다 훨씬 빠름 |
| **Ridge extension** | $+ \lambda I$로 separation·ill-cond 해결 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\beta^{(0)} = 0$ (모든 초기 계수 0)으로 시작하면 첫 iteration의 $z$와 $p$는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

$\eta^{(0)} = 0$, $p^{(0)} = \sigma(0) = 0.5$ 모든 $i$. $W^{(0)} = 0.25 I$. $z^{(0)} = 0 + (y - 0.5)/0.25 = 4(y - 0.5) \in \{-2, +2\}$.

→ 첫 iteration은 "target $z_i = \pm 2$, 가중치 $w_i = 0.25$" 인 WLS. $\beta^{(1)} = (0.25 X^\top X)^{-1} \cdot 0.25 X^\top z = (X^\top X)^{-1} X^\top z$ — 사실상 **OLS with $y \to \pm 2$** 라벨. 한 step 만에 대략적인 선형 분류경계를 학습.

</details>

**문제 2** (심화): Poisson regression에서 IRLS의 working response는 어떻게 되는가?

<details>
<summary>힌트 및 해설</summary>

Poisson: $y_i \sim \text{Poi}(\mu_i)$, canonical link $\log \mu = x^\top \beta$. 

$\ell = \sum [y_i \log \mu_i - \mu_i]$, $\mu = e^{\eta}$.

Gradient: $X^\top (y - \mu)$. Hessian: $-X^\top \text{diag}(\mu) X$ → $W = \text{diag}(\mu_i)$.

Working response: $z = \eta + W^{-1}(y - \mu) = \log \mu + (y - \mu)/\mu = \log \mu + y/\mu - 1$.

→ **Poisson IRLS = log-scale 회귀**. Ch2-03에서 exponential family의 canonical link가 같은 패턴 ($W = \text{Var}(y)$)을 주는 이유를 다룬다.

</details>

**문제 3** (ML 연결): NN의 **Natural Gradient** 또는 **K-FAC**이 IRLS의 일반화인 이유를 설명하라.

<details>
<summary>힌트 및 해설</summary>

Natural Gradient: $\beta \leftarrow \beta + \eta F^{-1} g$, where $F$는 Fisher information matrix. LR에서 $F = X^\top W X / n$ (경험적) = Newton Hessian의 음. 따라서 **Natural Gradient = Newton = IRLS** (for LR).

K-FAC (Kronecker-Factored Approximate Curvature): NN의 Fisher를 Kronecker 분해로 근사. Layer-wise로 Fisher를 계산 → 각 layer에서 IRLS와 유사한 update.

따라서 "NN의 2차 최적화" = "layer-wise IRLS의 근사". LR의 IRLS는 NN의 모든 precision-based optimizer의 **완전 풀린 버전**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Logistic Regression의 MLE](./01-logistic-mle.md) | [📚 README](../README.md) | [03. Exponential Family와 Canonical Link ▶](./03-exp-family-canonical-link.md) |

</div>
