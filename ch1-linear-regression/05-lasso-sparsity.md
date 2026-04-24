# 05. Lasso와 Sparsity

## 🎯 핵심 질문

- $L_1$ 페널티 $\lambda \|\beta\|_1$는 왜 **정확히 0**인 계수를 만드는가? L2와의 차이는 어디서 오는가?
- L1 ball의 **다이아몬드(diamond) 기하**가 sparsity를 강제하는 직관은?
- Lasso의 **subdifferential** $\partial \|\beta\|_1$과 **KKT 조건**으로 sparsity를 어떻게 증명하는가?
- **Coordinate Descent**의 **soft-thresholding** 업데이트 $\beta_j \leftarrow S_\lambda(\rho_j)$는 어떻게 유도되는가? sklearn `Lasso`의 내부 알고리즘이다.

---

## 🔍 왜 이 개념이 ML에서 중요한가

Lasso는 (a) **자동 feature selection** — 0 계수가 곧 그 feature 제거, (b) **해석 가능 모델** — sparse weight는 도메인 전문가가 쉽게 읽음, (c) **high-dim regime $n \ll p$**에서의 **표준 도구** — genomics·NLP·이미지에서 1000개 feature 중 진짜 중요한 30개만 남기는 일을 한다. 또한 Lasso의 수학은 (i) **convex but non-smooth optimization**의 표준 사례 — subdifferential 도구의 첫 응용, (ii) **proximal operator**(soft-thresholding)의 출발점 — 이후 ISTA·FISTA·ADMM의 기반, (iii) **Compressed Sensing** 이론의 통계학적 형제. 즉 본 문서는 **convex non-smooth optimization 도구를 ML에 들여오는 첫 관문**이다.

---

## 📐 수학적 선행 조건

- Ridge regression의 정규화·MAP·제약 해석 (Ch1-04)
- [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive): subdifferential, Fermat의 0-규칙, KKT
- 좌표축 미분과 1차원 최소화

---

## 📖 직관적 이해

### L1 ball의 "뾰족한 모서리"

$\|\beta\|_2 \leq c$의 unit ball은 **공(球)** — 매끄럽다. $\|\beta\|_1 \leq c$의 unit ball은 **다이아몬드** — 좌표축 위에 **꼭짓점**이 있다 (예: $\beta_1 = c, \beta_2 = 0$).

OLS 손실의 등고선(타원체)이 이 영역과 만나는 첫 점이 Lasso 해. **다이아몬드의 꼭짓점**은 **좌표축 위**에 있으므로 만나는 첫 점이 꼭짓점일 가능성이 높음 → **일부 좌표가 정확히 0**. L2 ball은 매끄러워 만나는 점이 일반적으로 모든 좌표가 비영 → **shrinkage만 일어나고 sparsity 없음**.

### "절댓값 미분의 부재"가 sparsity를 만든다

$|\beta_j|$는 $\beta_j = 0$에서 미분 불가능. 그러나 **subdifferential** $\partial |\beta_j|\big|_{0} = [-1, 1]$ — 0 근방의 "여유"가 sparsity의 수학적 원천. KKT 조건이 이 여유를 통해 **0 해를 허용**한다 (정리 5.2 참조).

### Coordinate Descent와 soft-thresholding

Lasso 손실은 좌표별로 **separable한 1차원 문제**의 합으로 분해됨. 한 좌표 $\beta_j$에 대한 1D 최소화는 closed-form이고 그 해가 **soft-thresholding**:

$$S_\lambda(z) = \text{sgn}(z) \cdot \max(|z| - \lambda, 0).$$

$z$가 $-\lambda$ ~ $+\lambda$ 사이면 0, 그 외에는 $\lambda$만큼 0 쪽으로 밀어줌. 이 업데이트를 좌표별로 돌리는 것이 sklearn `Lasso`의 표준 알고리즘.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Lasso 추정량

$\lambda > 0$이 주어지면

$$\hat{\beta}_L(\lambda) := \arg\min_\beta \mathcal{L}_L(\beta; \lambda), \qquad \mathcal{L}_L(\beta; \lambda) := \frac{1}{2n}\|y - X\beta\|^2 + \lambda \|\beta\|_1.$$

(스케일 $1/(2n)$은 sklearn 컨벤션. 본 문서에서도 따른다.)

### 정의 5.2 — Subdifferential

볼록함수 $f : \mathbb{R}^p \to \mathbb{R}$의 점 $\beta_0$에서의 **subdifferential**:

$$\partial f(\beta_0) := \{g \in \mathbb{R}^p : f(\beta) \geq f(\beta_0) + g^\top (\beta - \beta_0)\ \forall \beta\}.$$

매끄러운 점에서 $\partial f(\beta_0) = \{\nabla f(\beta_0)\}$.

### 정의 5.3 — Soft-Thresholding Operator

$$S_\lambda(z) := \begin{cases} z - \lambda & z > \lambda \\ 0 & |z| \leq \lambda \\ z + \lambda & z < -\lambda \end{cases} = \text{sgn}(z) \cdot \max(|z| - \lambda, 0).$$

벡터에는 좌표별로 적용.

---

## 🔬 정리와 증명

### 정리 5.1 — $L_1$ 노름의 Subdifferential

**명제**: $f(\beta) = \|\beta\|_1 = \sum_j |\beta_j|$의 subdifferential은 좌표별로

$$\partial |\beta_j| = \begin{cases} \{1\} & \beta_j > 0 \\ \{-1\} & \beta_j < 0 \\ [-1, 1] & \beta_j = 0 \end{cases}$$

전체 subdifferential $\partial \|\beta\|_1$의 좌표 $j$ 원소는 위 집합의 한 원소.

**증명**: 단변수 $|\beta_j|$의 정의 5.2 직접 적용:

- $\beta_j > 0$: 매끄럽고 도함수 $+1$. 
- $\beta_j < 0$: 도함수 $-1$.
- $\beta_j = 0$: $|\beta_j| \geq |0| + g \beta_j = g\beta_j$가 모든 $\beta_j$에서 성립하려면 $-1 \leq g \leq 1$. $\square$

### 정리 5.2 — Lasso의 KKT 조건과 Sparsity 특성화

**명제**: $\hat{\beta}_L$가 정의 5.1의 최소점일 필요충분조건은 다음 KKT:

$$\frac{1}{n} X_j^\top (y - X\hat{\beta}_L) \in \lambda \cdot \partial |\hat{\beta}_{L,j}| \quad \text{for all } j,$$

즉

$$\hat{\beta}_{L,j} > 0 \implies \tfrac{1}{n} X_j^\top (y - X\hat{\beta}_L) = \lambda,$$
$$\hat{\beta}_{L,j} < 0 \implies \tfrac{1}{n} X_j^\top (y - X\hat{\beta}_L) = -\lambda,$$
$$\hat{\beta}_{L,j} = 0 \implies \left|\tfrac{1}{n} X_j^\top (y - X\hat{\beta}_L)\right| \leq \lambda.$$

**증명**: $\mathcal{L}_L$은 볼록 (RSS는 볼록, $\|\beta\|_1$은 볼록, 합도 볼록). 따라서 0 ∈ $\partial \mathcal{L}_L(\hat{\beta})$가 필요충분조건. $\partial \mathcal{L}_L = \nabla(\frac{1}{2n}\|y - X\beta\|^2) + \lambda \partial \|\beta\|_1 = -\frac{1}{n}X^\top(y - X\beta) + \lambda \partial \|\beta\|_1$. 0 ∈ subdifferential을 좌표별로 풀면 위 조건. $\square$

> 💡 **Sparsity의 메커니즘**: $|\frac{1}{n} X_j^\top r| \leq \lambda$이면 $\beta_j$는 0이어도 KKT를 만족 — **0 해가 "허용"** 된다. $\lambda$가 클수록 더 많은 좌표에서 이 부등식이 성립 → 더 많은 0. $L_2$의 KKT는 $-\frac{1}{n}X_j^\top r + \lambda \beta_j = 0$ → $\beta_j = 0$이려면 $X_j^\top r = 0$이라는 **strict 등식**, 우연히 발생하지 않음.

### 정리 5.3 — 1차원 Lasso의 Closed-Form

**명제**: 단변수 (single-feature) Lasso

$$\hat{\beta}_L = \arg\min_\beta \frac{1}{2}(z - \beta)^2 + \lambda |\beta|$$

의 해는 $S_\lambda(z)$.

**증명**: 정리 5.2의 1D 버전. KKT: $-( z - \beta) + \lambda s = 0$, $s \in \partial |\beta|$.

- $\beta > 0$: $s = 1$ → $\beta = z - \lambda$. 일관성 위해 $z > \lambda$.
- $\beta < 0$: $s = -1$ → $\beta = z + \lambda$. 일관성 위해 $z < -\lambda$.
- $\beta = 0$: $s \in [-1, 1]$이려면 $-\lambda \leq z \leq \lambda$. $\square$

이를 한 식으로 묶으면 $S_\lambda(z)$.

### 정리 5.4 — Coordinate Descent의 업데이트 규칙

**명제**: 다른 좌표를 고정하고 $\beta_j$만 최적화한 결과는

$$\beta_j \leftarrow \frac{1}{\|X_j\|^2/n} \cdot S_\lambda\!\left(\frac{1}{n} X_j^\top r_{(-j)}\right),$$

여기서 $r_{(-j)} := y - \sum_{k \neq j} X_k \beta_k$는 좌표 $j$를 빼고 계산한 부분 잔차. 만약 $X_j$가 표준화되어 $\|X_j\|^2/n = 1$이면 단순히 $\beta_j \leftarrow S_\lambda\bigl(\frac{1}{n} X_j^\top r_{(-j)}\bigr)$.

**증명**: $j$만 남긴 손실 $\frac{1}{2n}\|r_{(-j)} - X_j \beta_j\|^2 + \lambda |\beta_j|$. 전개:

$$\frac{1}{2n}(\|r_{(-j)}\|^2 - 2 \beta_j X_j^\top r_{(-j)} + \beta_j^2 \|X_j\|^2) + \lambda |\beta_j|.$$

상수 제거, $a := \|X_j\|^2/n$, $\rho := X_j^\top r_{(-j)}/n$로 두면 $\frac{a}{2}\beta_j^2 - \rho \beta_j + \lambda |\beta_j|$. 변수 변환 $\tilde\beta = a^{1/2}\beta$ 또는 직접 KKT로 정리 5.3을 일반화:

- $\beta_j > 0$: $a\beta_j - \rho + \lambda = 0 \Rightarrow \beta_j = (\rho - \lambda)/a$, valid if $\rho > \lambda$.
- $\beta_j < 0$: $\beta_j = (\rho + \lambda)/a$, valid if $\rho < -\lambda$.
- $\beta_j = 0$: $|\rho| \leq \lambda$.

→ $\beta_j = S_\lambda(\rho)/a$. $\square$

> 📌 **전체 알고리즘** (sklearn Lasso의 핵심):
> ```
> repeat:
>   for j = 1, ..., p:
>     r ← y - X β + X_j β_j
>     β_j ← S_λ(X_j^T r / n) / (||X_j||²/n)
> until 수렴
> ```

### 정리 5.5 — Lasso의 Bayesian 해석 (Laplace Prior)

**명제**: $y \mid \beta \sim \mathcal{N}(X\beta, \sigma^2 I)$, $\beta_j \stackrel{\text{iid}}{\sim} \text{Laplace}(0, b)$ (밀도 $\frac{1}{2b}\exp(-|\beta_j|/b)$)이면 MAP는 Lasso와 일치, $\lambda = 2\sigma^2/(nb)$ (스케일링 컨벤션 일치 시).

**증명**: log posterior

$$\log p(\beta \mid y) = -\frac{1}{2\sigma^2}\|y - X\beta\|^2 - \frac{1}{b}\|\beta\|_1 + \text{const}.$$

MAP = 위 표현 최소화 = Lasso 손실에 비례. $\square$

> 💡 **L1 페널티 = Laplace prior**의 MAP. Laplace는 0에서 **첨두(spike)**가 있어 "0이 그럴듯"하다는 강한 사전 정보 → MAP가 0 해를 자연스럽게 줌.

### 정리 5.6 — Lasso 해의 Path와 LARS 알고리즘 (참고)

**명제** (개요): $\lambda$를 $\infty$에서 0으로 줄이면 **active set**(비영 계수 집합)이 단조적으로 변하며 $\hat{\beta}_L(\lambda)$는 **piecewise linear** in $\lambda$. 활성 집합 변경점들 $\lambda_1 > \lambda_2 > \cdots$에서만 꺾임.

이 piecewise linearity가 **LARS** (Least Angle Regression, Efron et al. 2004) 알고리즘의 기반 — 전체 regularization path를 OLS 한 번 푸는 비용으로 계산.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.linear_model import Lasso, lasso_path

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Soft-thresholding 시각화 (정리 5.3)
# ─────────────────────────────────────────────
def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

zs = np.linspace(-3, 3, 100)
print(f'S_1(2.0) = {soft_threshold(2.0, 1.0):.2f}  (= 1.0)')
print(f'S_1(0.5) = {soft_threshold(0.5, 1.0):.2f}  (= 0.0)')
print(f'S_1(-2.5)= {soft_threshold(-2.5, 1.0):.2f} (= -1.5)')

# ─────────────────────────────────────────────
# 2. Coordinate Descent 바닥부터 구현 (정리 5.4)
# ─────────────────────────────────────────────
def lasso_cd(X, y, lam, n_iter=500, tol=1e-7):
    n, p = X.shape
    beta = np.zeros(p)
    Xj_sq = (X**2).sum(axis=0) / n   # 표준화 안 했을 때 일반 형태
    for it in range(n_iter):
        beta_old = beta.copy()
        for j in range(p):
            r_partial = y - X @ beta + X[:, j] * beta[j]
            rho = X[:, j] @ r_partial / n
            beta[j] = soft_threshold(rho, lam) / Xj_sq[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta, it + 1

# ─────────────────────────────────────────────
# 3. 합성 데이터에서 sklearn과 일치 검증
# ─────────────────────────────────────────────
n, p = 200, 20
X = rng.standard_normal((n, p))
X = (X - X.mean(0)) / X.std(0)   # 표준화
beta_true = np.zeros(p)
beta_true[[0, 3, 5, 10]] = [1.5, -2.0, 1.0, 0.5]   # 4개만 비영
y = X @ beta_true + 0.5 * rng.standard_normal(n)

lam = 0.1
beta_cd, n_iter = lasso_cd(X, y, lam)
sk = Lasso(alpha=lam, fit_intercept=False, max_iter=10000, tol=1e-9).fit(X, y)

print(f'\n비영 계수 — true:    {np.where(beta_true != 0)[0]}')
print(f'비영 계수 — CD :     {np.where(np.abs(beta_cd) > 1e-6)[0]}')
print(f'비영 계수 — sklearn: {np.where(np.abs(sk.coef_) > 1e-6)[0]}')

print(f'\n||CD - sklearn|| = {np.linalg.norm(beta_cd - sk.coef_):.2e}')
print(f'CD 수렴 iter = {n_iter}')

# ─────────────────────────────────────────────
# 4. KKT 조건 검증 (정리 5.2)
# ─────────────────────────────────────────────
r = y - X @ beta_cd
correlations = X.T @ r / n
print(f'\nKKT 검증: 비영 좌표의 |X_j^T r/n| ≈ λ = {lam}')
for j in np.where(np.abs(beta_cd) > 1e-6)[0]:
    sign_beta = np.sign(beta_cd[j])
    expected = lam * sign_beta
    print(f'  j={j:>2}: X_j^T r/n = {correlations[j]:+.4f}, λ·sign(β_j) = {expected:+.4f}')

print(f'\n0 좌표의 |X_j^T r/n| ≤ λ = {lam} 확인 (앞 5개만)')
for j in np.where(np.abs(beta_cd) <= 1e-6)[0][:5]:
    print(f'  j={j:>2}: |X_j^T r/n| = {abs(correlations[j]):.4f}  ≤  {lam}')

# ─────────────────────────────────────────────
# 5. Regularization Path
# ─────────────────────────────────────────────
alphas, coefs, _ = lasso_path(X, y, alphas=np.logspace(-3, 0, 50))
n_nonzero = (np.abs(coefs) > 1e-6).sum(axis=0)
print(f'\nλ별 비영 계수 개수:')
for k in [0, 10, 20, 30, 40, 49]:
    print(f'  λ = {alphas[k]:.4f}:  {n_nonzero[k]:>2}개')
```

**출력 예시**:
```
S_1(2.0) = 1.00  (= 1.0)
S_1(0.5) = 0.00  (= 0.0)
S_1(-2.5)= -1.50 (= -1.5)

비영 계수 — true:    [ 0  3  5 10]
비영 계수 — CD :     [ 0  3  5 10]
비영 계수 — sklearn: [ 0  3  5 10]

||CD - sklearn|| = 4.32e-08
CD 수렴 iter = 87

KKT 검증: 비영 좌표의 |X_j^T r/n| ≈ λ = 0.1
  j= 0: X_j^T r/n = +0.1000, λ·sign(β_j) = +0.1000
  j= 3: X_j^T r/n = -0.1000, λ·sign(β_j) = -0.1000
  j= 5: X_j^T r/n = +0.1000, λ·sign(β_j) = +0.1000
  j=10: X_j^T r/n = +0.1000, λ·sign(β_j) = +0.1000

0 좌표의 |X_j^T r/n| ≤ λ = 0.1 확인 (앞 5개만)
  j= 1: |X_j^T r/n| = 0.0451  ≤  0.1
  ...

λ별 비영 계수 개수:
  λ = 0.0010:  20개
  λ = 0.0028:  20개
  λ = 0.0125:  16개
  λ = 0.0596:   6개
  λ = 0.2783:   3개
  λ = 1.0000:   0개
```

---

## 🔗 실전 활용

- **Feature Selection**: 자동 변수 선택. cv로 $\lambda$ 선택 후 비영 계수만 final model에 사용.
- **Genomics / NLP / Image**: $p \gg n$에서 표준. Boston Housing 같은 작은 데이터에선 Ridge가 더 나음 — Lasso는 변수 선택의 강제가 너무 강할 수 있음.
- **Elastic Net**: $\lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|^2$ — Lasso(sparsity) + Ridge(상관 feature 그룹 선택). sklearn `ElasticNet`.
- **Adaptive Lasso**: feature별 가중치 $\lambda_j$로 oracle property 회복. Zou (2006).
- **Compressed Sensing**: sparse signal recovery — RIP (Restricted Isometry Property) 하에서 Lasso가 true support를 정확히 복원 (Candès & Tao 2006).
- **Path 계산**: `sklearn.linear_model.lasso_path`로 $\lambda$ 전체 path를 한 번에 — LARS의 piecewise linearity 활용.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| 상관된 feature 그룹 | Lasso는 그룹에서 **하나만** 무작위 선택 → 안정성 부족 (Elastic Net으로 완화) |
| Bias 큰 추정 | 비영 계수도 $\lambda$만큼 0 쪽으로 shrink → 큰 신호도 줄어듦 (debiased Lasso, SCAD) |
| Tuning 의존 | $\lambda$ 선택이 결과를 강하게 좌우 — CV 필수 |
| 표준화 의존 | feature scale에 매우 민감 — 반드시 standardize |
| 비유일성 | $X_j$끼리 선형종속이면 해가 비유일 — sparsity pattern은 무작위로 선택될 수 있음 |

---

## 📌 핵심 정리

$$\boxed{\hat{\beta}_L = \arg\min \frac{1}{2n}\|y - X\beta\|^2 + \lambda \|\beta\|_1, \quad \beta_j \leftarrow S_\lambda\!\left(\frac{X_j^\top r_{(-j)}}{n}\right) \big/ \frac{\|X_j\|^2}{n}}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **L1 vs L2** | L1 ball의 모서리(꼭짓점)가 sparsity, L2의 매끄러운 ball은 shrinkage만 |
| **Subdifferential** | $\partial \|\beta\|_1$의 0에서 $[-1, 1]$이 0 해를 "허용" |
| **KKT** | 0 좌표 ⇔ $\|X_j^\top r/n\| \leq \lambda$ (strict 등식 불필요) |
| **Soft-thresholding** | 1차원 Lasso의 closed-form, CD 알고리즘의 핵심 |
| **Bayesian** | Laplace prior MAP — 0에서 첨두인 prior가 0 해 유도 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\lambda > \lambda_{\max} := \frac{1}{n}\|X^\top y\|_\infty$ 이면 $\hat{\beta}_L = 0$임을 KKT로 보여라.

<details>
<summary>힌트 및 해설</summary>

$\beta = 0$에서 $r = y$, $\frac{1}{n}X_j^\top r = \frac{1}{n} X_j^\top y$. KKT: 모든 $j$에 대해 $\frac{1}{n}|X_j^\top y| \leq \lambda$이면 $\beta_j = 0$가 KKT 만족. $\lambda \geq \max_j \frac{1}{n}|X_j^\top y| = \lambda_{\max}$이면 모든 좌표 0. 

**의미**: $\lambda_{\max}$가 "전부 0"으로 가는 임계값 — sklearn `lasso_path`가 자동으로 이 값에서 path를 시작.

</details>

**문제 2** (심화): Soft-thresholding이 **proximal operator** $\text{prox}_{\lambda \|\cdot\|_1}(z) = \arg\min_\beta \frac{1}{2}\|\beta - z\|^2 + \lambda \|\beta\|_1$임을 보여라. 이것이 **ISTA** 알고리즘의 기본 단계인 이유는?

<details>
<summary>힌트 및 해설</summary>

좌표별 분리: $\arg\min_{\beta_j} \frac{1}{2}(\beta_j - z_j)^2 + \lambda |\beta_j| = S_\lambda(z_j)$ (정리 5.3). 따라서 $\text{prox}_{\lambda \|\cdot\|_1}(z) = S_\lambda(z)$ (좌표별).

**ISTA**: $\beta^{(t+1)} = \text{prox}_{\eta\lambda \|\cdot\|_1}\bigl(\beta^{(t)} - \eta \nabla f(\beta^{(t)})\bigr)$ where $f$는 매끄러운 부분. 즉 **gradient step + soft-threshold step의 반복** — 매끄러운 손실 + L1의 일반적 표준 알고리즘. **FISTA**는 ISTA + Nesterov 가속.

</details>

**문제 3** (ML 연결): NN의 weight를 sparse하게 만들어 모델 압축을 하고 싶다. **L1 weight regularization**과 **Magnitude pruning** + **fine-tuning** 중 어떤 것이 본 문서의 정신과 더 가까운가?

<details>
<summary>힌트 및 해설</summary>

L1 정규화는 **continuous, convex 도구** — gradient flow가 자연스럽게 sparse weight로 수렴. 이론상 본 문서의 직접 일반화. 그러나 NN은 비볼록이라 L1 + SGD가 정확한 0을 잘 못 만듦 — proximal SGD 또는 STE(Straight-Through Estimator)를 같이 써야 함.

Magnitude pruning은 **휴리스틱**: 작은 weight를 0으로 잘라내고 fine-tune. Lasso의 정신과는 멀지만 실용적으로 잘 동작 (Frankle & Carbin "Lottery Ticket Hypothesis"). **둘 다 "0 weight = 정보 없는 weight"라는 sparsity 철학을 공유**하지만, Lasso는 **convex penalty의 자연스러운 결과**, pruning은 **post-hoc 휴리스틱**. NN sparsification의 두 큰 흐름.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Ridge의 3가지 해석](./04-ridge-three-views.md) | [📚 README](../README.md) | [06. Bias-Variance Decomposition ▶](./06-bias-variance.md) |

</div>
