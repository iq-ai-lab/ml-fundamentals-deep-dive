# 02. 기하학적 관점 — 수직투영

## 🎯 핵심 질문

- $\hat{y} = X\hat{\beta}$가 왜 $y$의 $\text{col}(X)$로의 **수직투영**인가?
- "잔차 $r = y - \hat{y}$가 $X$의 열에 수직" — 이 한 문장이 왜 정규방정식 $X^\top(y - X\beta) = 0$과 정확히 같은가?
- **Hat matrix** $H = X(X^\top X)^{-1} X^\top$는 왜 idempotent($H^2 = H$)이고 대칭인가? leverage $h_{ii}$의 통계적 의미는?
- 무한차원 Hilbert 공간으로 일반화하면 RKHS의 함수 회귀와 어떻게 연결되는가?

---

## 🔍 왜 이 관점이 ML에서 중요한가

MLE 유도(Ch1-01)는 "왜 그 식이 나오는가"의 **확률적 답**을, 본 문서의 기하학적 관점은 **공간적 답**을 준다. 같은 식 $\hat{\beta} = (X^\top X)^{-1} X^\top y$를 두 시점에서 보면 (a) **잡음 가정 없이도** 식이 자연스러움을 알게 되고, (b) leverage·influence·outlier 같은 진단 도구의 기하학적 의미가 명확해지며, (c) 이 관점은 그대로 **Hilbert 공간** $\mathcal{H}$에서의 함수 회귀로 일반화되어 **Kernel Ridge Regression·Gaussian Process posterior mean**의 통합 시점이 된다. "투영"이라는 한 단어가 OLS, Ridge, Lasso(부분), KRR, GP, MLP 마지막 layer를 모두 묶는다.

---

## 📐 수학적 선행 조건

- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 부분공간·기저, 직교여공간(orthogonal complement), 사영 행렬
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): Hilbert 공간의 수직분해 정리(Projection Theorem)
- 미적분: 1계 조건의 기하학적 해석 (gradient ⊥ level set)

---

## 📖 직관적 이해

### "최소거리"는 본질적으로 "수직"이다

$y \in \mathbb{R}^n$이 한 점, $\text{col}(X) = \{X\beta : \beta \in \mathbb{R}^p\}$이 한 부분공간(평면). 부분공간 위의 점 중 $y$에 가장 가까운 점이 $\hat{y}$. 초등 기하에서 이미 익숙하다: **$y$에서 평면으로의 최단 거리는 수직선의 발(foot of perpendicular)** 이다. 곧

$$\hat{y} = \text{Proj}_{\text{col}(X)}(y), \qquad y - \hat{y} \perp \text{col}(X).$$

### 수직성 = 정규방정식

"$y - \hat{y}$가 $\text{col}(X)$의 모든 벡터에 수직"을 식으로 쓰면 임의의 $v = X\alpha$에 대해 $v^\top (y - \hat{y}) = 0$, 즉 $\alpha^\top X^\top (y - X\hat{\beta}) = 0$이 모든 $\alpha$에서 성립. 따라서

$$X^\top (y - X\hat{\beta}) = 0 \iff X^\top X \hat{\beta} = X^\top y.$$

이것이 바로 Ch1-01에서 미분으로 얻은 정규방정식. **확률적 유도와 기하학적 유도가 정확히 같은 식**에 도달한다.

### Hat matrix는 "투영자(projector)"

$\hat{y} = X\hat{\beta} = X (X^\top X)^{-1} X^\top y =: Hy$. 행렬 $H$는 **임의 벡터 $y$를 $\text{col}(X)$로 사영**하는 일을 한다. "y에 hat을 씌운다"는 의미로 **Hat matrix**라 부른다. 사영자의 두 가지 핵심 성질은:

1. **Idempotent**: $H^2 = H$ — 한 번 사영한 점을 다시 사영하면 자기 자신.
2. **대칭(symmetric)**: $H^\top = H$ — orthogonal projection임을 보장.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Column Space와 Orthogonal Complement

$X \in \mathbb{R}^{n \times p}$의 **column space** $\text{col}(X) := \{X\beta : \beta \in \mathbb{R}^p\} \subset \mathbb{R}^n$. 그 **orthogonal complement** $\text{col}(X)^\perp := \{u \in \mathbb{R}^n : u^\top v = 0 \ \forall v \in \text{col}(X)\}$. $\mathbb{R}^n = \text{col}(X) \oplus \text{col}(X)^\perp$.

### 정의 2.2 — Orthogonal Projection

폐쇄된 부분공간 $V \subset \mathbb{R}^n$로의 **orthogonal projection** $P_V : \mathbb{R}^n \to V$는 다음을 만족하는 유일한 선형 사상:

$$P_V y \in V, \qquad y - P_V y \in V^\perp.$$

행렬로 표현하면 $P_V$는 (i) $P_V^2 = P_V$, (ii) $P_V^\top = P_V$를 만족한다. 거꾸로 이 두 조건을 만족하는 행렬은 어떤 부분공간으로의 orthogonal projection이다.

### 정의 2.3 — Hat Matrix와 Annihilator

$X$가 full column rank이면

$$H := X (X^\top X)^{-1} X^\top \in \mathbb{R}^{n \times n}$$

을 **hat matrix** (또는 projection matrix)라 한다. $M := I - H$를 **annihilator** (residual maker)라 한다.

### 정의 2.4 — Leverage

Hat matrix의 대각원소 $h_{ii} = [H]_{ii}$를 **$i$번째 점의 leverage**라 한다. $h_{ii} \in [0, 1]$이고 $\sum_i h_{ii} = \text{tr}(H) = p$.

---

## 🔬 정리와 증명

### 정리 2.1 — 수직분해 정리 (유한차원판 Projection Theorem)

**명제**: $V \subset \mathbb{R}^n$이 부분공간이면 임의의 $y \in \mathbb{R}^n$에 대해 다음을 만족하는 유일한 분해 $y = y_V + y_{V^\perp}$가 존재한다:

$$y_V \in V, \quad y_{V^\perp} \in V^\perp, \quad \|y - v\|^2 \geq \|y_{V^\perp}\|^2 \ \forall v \in V \quad (\text{등호는 } v = y_V).$$

**증명**: $V$의 정규직교기저 $\{e_1, \ldots, e_k\}$ ($k = \dim V$)를 잡으면 $y_V := \sum_{i=1}^k \langle y, e_i \rangle e_i$로 정의. 임의의 $v = \sum c_i e_i \in V$에 대해

$$\|y - v\|^2 = \|y\|^2 - 2\sum_i c_i \langle y, e_i \rangle + \sum_i c_i^2$$

는 $c_i = \langle y, e_i\rangle$일 때 최소이고 그 값은 $\|y\|^2 - \sum \langle y, e_i \rangle^2 = \|y - y_V\|^2$. $y_{V^\perp} := y - y_V$가 $V^\perp$에 속함은 $\langle y_{V^\perp}, e_j \rangle = \langle y, e_j \rangle - \sum_i \langle y, e_i\rangle \delta_{ij} = 0$. 유일성은 $V \cap V^\perp = \{0\}$에서. $\square$

### 정리 2.2 — Normal Equation의 기하학적 유도

**명제**: $\hat{\beta}$가 $\|y - X\beta\|^2$를 최소화 $\iff$ $X^\top (y - X\hat{\beta}) = 0$.

**증명**: 정리 2.1을 $V = \text{col}(X)$에 적용하면 최소점 $\hat{y} \in V$는 $y - \hat{y} \in V^\perp$를 만족하는 유일한 점. $\hat{y} = X\hat{\beta}$로 표현하면 $y - X\hat{\beta} \perp \text{col}(X)$, 즉 모든 $v \in \text{col}(X)$에 대해 $v^\top (y - X\hat{\beta}) = 0$. 이는 $V$의 모든 기저 $X$의 열 $x_{(j)}$에 대해 $x_{(j)}^\top (y - X\hat{\beta}) = 0$, 행렬로 묶으면 $X^\top (y - X\hat{\beta}) = 0$. $\square$

### 정리 2.3 — Hat Matrix의 성질

**명제**: $X$가 full column rank이면 $H = X (X^\top X)^{-1} X^\top$는 다음을 만족:

1. **Idempotent**: $H^2 = H$.
2. **대칭**: $H^\top = H$.
3. **Range**: $\text{col}(H) = \text{col}(X)$.
4. **고유값**: $H$의 고유값은 $\{0, 1\}$ — 1의 중복도는 $p$, 0의 중복도는 $n - p$.
5. **Trace**: $\text{tr}(H) = p$ — 모델 자유도와 일치.

**증명**:

(1) $H^2 = X(X^\top X)^{-1} X^\top X (X^\top X)^{-1} X^\top = X (X^\top X)^{-1} X^\top = H$.

(2) $H^\top = X^\top{}^\top ((X^\top X)^{-1})^\top X^\top = X (X^\top X)^{-1} X^\top = H$ ($X^\top X$는 대칭이므로 역도 대칭).

(3) $\text{col}(H) = H \cdot \mathbb{R}^n \subseteq \text{col}(X)$ (자명). 역으로 $v = X\beta \in \text{col}(X)$이면 $H v = X(X^\top X)^{-1} X^\top X \beta = X\beta = v$. 따라서 $\text{col}(X) \subseteq \text{col}(H)$.

(4) $H$는 대칭 idempotent → 고유값 분해 $H = Q \Lambda Q^\top$. $H^2 = H \Rightarrow \Lambda^2 = \Lambda \Rightarrow \lambda_i \in \{0, 1\}$. 1-고유공간 = $\text{col}(H) = \text{col}(X)$로 차원 $p$, 0-고유공간 = $\text{col}(X)^\perp$로 차원 $n - p$.

(5) $\text{tr}(H) = \sum \lambda_i = p \cdot 1 + (n - p) \cdot 0 = p$. $\square$

### 정리 2.4 — Annihilator $M = I - H$

**명제**: $M = I - H$는 $\text{col}(X)^\perp$로의 orthogonal projection, 다음을 만족:

1. $M^2 = M$, $M^\top = M$.
2. $MX = 0$ (이름 "annihilator"의 유래).
3. $My = $ 잔차 $r = y - \hat{y}$.
4. $\text{tr}(M) = n - p$ (잔차 자유도).

**증명**: (1) $H$가 사영자이면 $I - H$도 사영자. (2) $MX = (I - H)X = X - HX = X - X = 0$ ($HX = X$는 $H$가 $\text{col}(X)$ 위에서 항등). (3) $My = y - Hy = y - \hat{y} = r$. (4) $\text{tr}(M) = n - p$. $\square$

> 💡 **"잔차 자유도"**: $r = My$, $r$은 $\text{col}(X)^\perp$의 원소 — 그 공간 차원이 $n - p$이므로 잔차는 사실상 $n - p$개의 자유도만 가진다. 이것이 $\sigma^2$ 비편향 추정이 $\mathrm{RSS}/(n - p)$인 이유 (Ch1-01 정리 1.5와 정확히 일치).

### 정리 2.5 — Leverage의 통계적 의미

**명제**: 모델 1.1 하에서

1. $\text{Var}(\hat{y}_i) = \sigma^2 h_{ii}$.
2. $\text{Var}(r_i) = \sigma^2 (1 - h_{ii})$.
3. $h_{ii} \in [0, 1]$이고 $\sum h_{ii} = p$.

따라서 $h_{ii}$가 1에 가까울수록 그 점의 예측은 자기 자신 $y_i$에 의해 결정 → **고-leverage 점**.

**증명**: $\hat{y} = Hy = HX\beta + H\epsilon$이고 $HX\beta = X\beta$이므로 $\hat{y} = X\beta + H\epsilon$. 따라서 $\text{Var}(\hat{y}) = H \cdot \sigma^2 I \cdot H^\top = \sigma^2 H$ ($H = H^\top$, $H^2 = H$). 대각원소가 $\sigma^2 h_{ii}$. $r = (I - H)\epsilon$ 같은 방식으로 $\text{Var}(r) = \sigma^2 (I - H)$, 대각이 $\sigma^2(1 - h_{ii})$. 합 $\sum h_{ii} = \text{tr}(H) = p$. 범위 $h_{ii} \in [0, 1]$은 정리 2.3의 고유값 분해와 $H$의 PSD성에서. $\square$

> 📌 **회귀 진단 규칙**: $h_{ii} > 2p/n$이면 "high leverage point"로 의심. **Cook's distance** $D_i = \frac{r_i^2}{p \hat{\sigma}^2} \cdot \frac{h_{ii}}{(1 - h_{ii})^2}$로 영향력 측정.

### 정리 2.6 — Hilbert 공간으로의 일반화

**명제**: $(\mathcal{H}, \langle \cdot, \cdot \rangle)$이 Hilbert 공간, $V \subset \mathcal{H}$가 폐쇄된 부분공간이면 임의의 $y \in \mathcal{H}$에 대해 유일한 $y_V \in V$가 존재해

$$\|y - y_V\| = \min_{v \in V} \|y - v\|, \qquad y - y_V \in V^\perp.$$

**증명 스케치**: 거리함수 $d(y, V) = \inf_{v \in V} \|y - v\|$로 minimizing sequence $\{v_n\}$를 잡으면 평행사변형 법칙으로 Cauchy임을 보이고, $\mathcal{H}$가 완비이므로 극한이 존재. 폐쇄성으로 극한 $\in V$. 수직성은 변분 (variation) $v_V + tw$로 $t \to 0$ 미분. $\square$

> 💡 **연결**: Kernel Ridge Regression의 Representer 정리 ($f^* = \sum \alpha_i k(\cdot, x_i)$)는 본질적으로 RKHS $\mathcal{H}_k$에서의 정리 2.6의 사례. **OLS = $\mathbb{R}^n$의 사영, KRR = $\mathcal{H}_k$의 사영**, 같은 정리의 두 차원.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Hat matrix의 idempotent·대칭·range 확인
# ─────────────────────────────────────────────
n, p = 50, 4
X = rng.standard_normal((n, p))
H = X @ np.linalg.solve(X.T @ X, X.T)

print(f'||H^2 - H||_F  = {np.linalg.norm(H @ H - H):.2e}    (idempotent)')
print(f'||H^T - H||_F  = {np.linalg.norm(H.T - H):.2e}      (symmetric)')

eigs = np.linalg.eigvalsh(H)
print(f'eigenvalues 분포: 1의 개수 = {np.sum(eigs > 0.5)} (= p = {p})')
print(f'                  0의 개수 = {np.sum(eigs < 0.5)} (= n-p = {n-p})')
print(f'tr(H) = {np.trace(H):.4f}  (= p)')

# ─────────────────────────────────────────────
# 2. 정규방정식 = 잔차 수직성
# ─────────────────────────────────────────────
beta_true = rng.standard_normal(p)
y = X @ beta_true + 0.1 * rng.standard_normal(n)
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
r = y - X @ beta_hat

print(f'\nmax |X^T r| = {np.max(np.abs(X.T @ r)):.2e}  (≈ 0이어야 — 수직성)')

# ─────────────────────────────────────────────
# 3. M = I - H가 잔차를 만든다 (annihilator)
# ─────────────────────────────────────────────
M = np.eye(n) - H
print(f'||M y - r||  = {np.linalg.norm(M @ y - r):.2e}')
print(f'||M X||      = {np.linalg.norm(M @ X):.2e}  (annihilate X)')

# ─────────────────────────────────────────────
# 4. Leverage h_ii의 분산 검증 (정리 2.5)
# ─────────────────────────────────────────────
sigma = 0.5
n_trials = 3000
y_hat_samples = np.zeros((n_trials, n))
r_samples = np.zeros((n_trials, n))
for t in range(n_trials):
    eps = sigma * rng.standard_normal(n)
    y_t = X @ beta_true + eps
    bh = np.linalg.solve(X.T @ X, X.T @ y_t)
    y_hat_samples[t] = X @ bh
    r_samples[t] = y_t - y_hat_samples[t]

h_diag = np.diag(H)
emp_var_yhat = y_hat_samples.var(axis=0)
emp_var_r    = r_samples.var(axis=0)

# 첫 5개 점만 비교
for i in range(5):
    print(f'  i={i}: h_ii={h_diag[i]:.3f}  '
          f'Var(ŷ_i): emp={emp_var_yhat[i]:.4f}  theo={sigma**2 * h_diag[i]:.4f}  '
          f'Var(r_i): emp={emp_var_r[i]:.4f}  theo={sigma**2 * (1-h_diag[i]):.4f}')

# ─────────────────────────────────────────────
# 5. 시각화: 2D에서 y → col(X) 사영
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
X2 = np.array([[1, 2], [2, 0.5], [0.5, 1], [1.5, 1.8]])  # 2 features만 (시각화)
y2 = np.array([3.0, 1.0, 1.5, 4.0])
H2 = X2 @ np.linalg.solve(X2.T @ X2, X2.T)
yhat2 = H2 @ y2
# 그림에 vector 표시 — 코드 생략 (개념 확인용)
print(f'\n2D 예시: ||y - ŷ|| = {np.linalg.norm(y2 - yhat2):.4f}')
print(f'        X^T(y - ŷ) = {X2.T @ (y2 - yhat2)}  (≈ 0)')
```

**출력 예시**:
```
||H^2 - H||_F  = 1.18e-15    (idempotent)
||H^T - H||_F  = 0.00e+00    (symmetric)
eigenvalues 분포: 1의 개수 = 4 (= p = 4)
                  0의 개수 = 46 (= n-p = 46)
tr(H) = 4.0000  (= p)

max |X^T r| = 5.96e-15  (≈ 0이어야 — 수직성)
||M y - r||  = 4.21e-15
||M X||      = 8.74e-15  (annihilate X)

  i=0: h_ii=0.072  Var(ŷ_i): emp=0.018  theo=0.018  Var(r_i): emp=0.234  theo=0.232
  i=1: h_ii=0.165  Var(ŷ_i): emp=0.041  theo=0.041  Var(r_i): emp=0.207  theo=0.209
  ...
```

---

## 🔗 실전 활용

- **회귀 진단**: $h_{ii} > 2p/n$인 점은 high leverage. $h_{ii} \to 1$이면 그 점이 회귀선을 결정.
- **Cook's distance·DFFITS**: leverage와 잔차를 결합해 점별 영향력 측정.
- **자유도 보정**: 잔차 분산 추정에 $1/(n - p)$가 들어가는 이유가 $\dim \text{col}(X)^\perp = n - p$임에서.
- **Anova의 직교분해**: $\|y - \bar{y}\|^2 = \|y - \hat{y}\|^2 + \|\hat{y} - \bar{y}\|^2$ — Pythagoras 정리의 통계 버전, $R^2$의 정의 근거.
- **GLS·WLS 일반화**: 잡음 공분산 $\Omega$가 $I$가 아니면 $\langle u, v \rangle_\Omega := u^\top \Omega^{-1} v$ 내적으로 같은 사영 구조.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $X$ full column rank | 비유일 사영 → Pseudoinverse(Ch1-03) |
| 표준 내적 $\langle u, v \rangle = u^\top v$ | 비등분산이면 GLS의 $\Omega^{-1}$-내적으로 일반화 |
| 유한차원 $\mathbb{R}^n$ | RKHS / GP에서는 무한차원 사영 (정리 2.6) |

**주의**: $H$의 idempotency는 "한 번 사영한 것을 또 사영하면 그대로"라는 뜻. 비-orthogonal projection (예: oblique projection)은 idempotent이지만 **대칭이 아닐 수 있다**. OLS의 $H$는 두 성질을 모두 갖춘 **orthogonal projection**.

---

## 📌 핵심 정리

$$\boxed{\hat{y} = Hy,\ H = X(X^\top X)^{-1}X^\top,\ H^2 = H,\ H^\top = H,\ (I - H)y = r \perp \text{col}(X)}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **수직성 = Normal Eq** | $r \perp \text{col}(X) \iff X^\top r = 0 \iff X^\top X \hat{\beta} = X^\top y$ |
| **Hat matrix** | $\text{col}(X)$로의 orthogonal projection, 고유값 $\{0, 1\}$ |
| **Annihilator** | $M = I - H$, $MX = 0$, $My = r$, $\text{tr}(M) = n - p$ |
| **Leverage** | $h_{ii} = \text{Var}(\hat{y}_i)/\sigma^2$, 합 = $p$ |
| **Hilbert 일반화** | 같은 정리가 RKHS·GP·KRR의 사영 구조를 만든다 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $X = \mathbf{1}_n$일 때 $H$의 모양을 직접 써라. $h_{ii}$는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

$X^\top X = n$, $H = \mathbf{1}\mathbf{1}^\top / n$ — 모든 원소가 $1/n$인 $n \times n$ 행렬. $h_{ii} = 1/n$, $\sum h_{ii} = 1 = p$. $\hat{y}_i = (Hy)_i = \sum_j y_j / n = \bar{y}$ — 모두가 표본평균으로 사영.

**기하학적 그림**: $\text{col}(X)$는 1차원 직선 $\{c \mathbf{1} : c \in \mathbb{R}\}$. 임의 $y$를 이 직선으로 사영하면 $\bar{y} \mathbf{1}$.

</details>

**문제 2** (심화): $X$의 두 열이 평행 ($x_{(2)} = c x_{(1)}$)일 때 $X^\top X$가 특이임을 보이고, 이때 무엇이 비유일해지는가?

<details>
<summary>힌트 및 해설</summary>

$X^\top X = \begin{pmatrix} \|x_{(1)}\|^2 & c\|x_{(1)}\|^2 \\ c\|x_{(1)}\|^2 & c^2\|x_{(1)}\|^2 \end{pmatrix}$ — 두 행이 비례, $\det = 0$. 따라서 $\hat{\beta}$는 비유일.

**중요**: $\hat{y} = X\hat{\beta}$ (사영) 자체는 여전히 **유일** — column space에 대한 사영은 $X$의 표현(parameterization)에 무관. 다만 $\beta$ 좌표가 $\beta_2 = c\beta_1 + \text{const}$ 직선 위 모든 점에서 같은 $\hat{y}$를 준다.

**해결**: Pseudoinverse는 이 직선 위 **min-norm 해**를 고르고 (Ch1-03), Ridge는 $\lambda I$를 더해 가역화 + 유일화 (Ch1-04).

</details>

**문제 3** (ML 연결): Gaussian Process posterior mean이 $m_*(x) = k(x, X)(K + \sigma^2 I)^{-1} y$라는 것을 본문 정리 2.6과 어떻게 연결할 수 있는가?

<details>
<summary>힌트 및 해설</summary>

GP posterior mean은 RKHS $\mathcal{H}_k$에서 $\sum L(y_i, f(x_i)) + \lambda \|f\|^2$의 최소화 해와 정확히 일치 (Kernel Methods 레포 Ch4-03). Representer 정리에 의해 해는 $f^* = \sum \alpha_i k(\cdot, x_i) \in \text{span}\{k(\cdot, x_i)\}$. 이는 **RKHS의 한 부분공간으로의 사영** — 정리 2.6의 RKHS 버전. $\mathbb{R}^n$의 OLS와 RKHS의 KRR은 **같은 사영 정리의 두 사례**. "선형 회귀를 기하학적으로 이해하면 GP/KRR의 절반은 끝난다."

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. MLE 관점에서의 선형 회귀](./01-mle-derivation.md) | [📚 README](../README.md) | [03. Moore-Penrose Pseudoinverse ▶](./03-pseudoinverse.md) |

</div>
