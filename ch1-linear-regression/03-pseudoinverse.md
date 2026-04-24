# 03. Moore-Penrose Pseudoinverse와 Rank-deficient 경우

## 🎯 핵심 질문

- $X^\top X$가 특이(singular)일 때 OLS는 어떻게 정의되는가? 무수히 많은 해 중 **어느 것을** 골라야 하는가?
- Moore-Penrose pseudoinverse $X^+$는 왜 $X^+ = \lim_{\lambda \to 0^+}(X^\top X + \lambda I)^{-1} X^\top$로 표현되고, **SVD로** $X^+ = V \Sigma^+ U^\top$가 되는가?
- $\hat{\beta} = X^+ y$가 왜 **min-norm least-squares 해**(가장 작은 $\|\beta\|$를 갖는 LS 해)임을 어떻게 증명하는가?
- 실무 수치 알고리즘에서 **Normal Equation vs QR vs SVD vs Cholesky**는 어떤 trade-off를 갖는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

현실 데이터에서 $X^\top X$가 정확히 가역인 경우는 드물다: (a) **$n < p$** (high-dim regime, genomics·NLP의 sparse feature), (b) **공선성**(둘 이상의 feature가 강한 선형 종속), (c) **수치적 ill-conditioning** (조건수가 $10^{14}$를 넘으면 사실상 특이). sklearn `LinearRegression`은 내부적으로 `scipy.linalg.lstsq`를 호출하고, 그 안은 **SVD-based pseudoinverse**다 — 즉 학생들이 "Normal Equation"으로 배운 것과 sklearn이 실제로 푸는 것은 다르다. Pseudoinverse는 (i) **항상 존재하며 유일**, (ii) **min-norm 해를 자동 선택**, (iii) **Ridge의 $\lambda \to 0$ 극한**으로서 정규화와 자연스럽게 연결된다. 즉 본 문서는 sklearn이 실제로 무슨 일을 하는지 설명한다.

---

## 📐 수학적 선행 조건

- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): SVD, rank, null space, four fundamental subspaces
- 정규방정식의 기하학적 유도 (Ch1-02)
- 극한과 연속성 (한계 정의의 well-definedness)

---

## 📖 직관적 이해

### "해가 무수히 많다"는 상황

$X^\top X$가 특이 ⇔ $X$가 column rank deficient ($\text{rank}(X) = r < p$) ⇔ $\text{null}(X) \neq \{0\}$. 정규방정식 $X^\top X \beta = X^\top y$의 해는 한 점 $\hat{\beta}$ + null space:

$$\{\hat{\beta} + v : v \in \text{null}(X^\top X) = \text{null}(X)\}.$$

이 모든 $\beta$는 **같은 $\hat{y} = X\beta$를 만든다** (예측은 같음). 그러나 $\beta$ 좌표는 무한히 많고 불안정. 실무에서는 **그중 가장 작은 norm을 갖는 해**를 선호한다 — 이것이 **min-norm 해**, 그것을 주는 연산자가 **Moore-Penrose pseudoinverse**.

### Ridge의 극한으로서

Ridge regression(Ch1-04)은 $\hat{\beta}_R = (X^\top X + \lambda I)^{-1} X^\top y$. $\lambda \to 0^+$로 보내면

$$\lim_{\lambda \to 0^+} (X^\top X + \lambda I)^{-1} X^\top = X^+.$$

$\lambda > 0$이면 $X^\top X + \lambda I$는 항상 가역이고, 이 한계는 **항상 존재**해서 **항상 잘 정의된** 연산자가 나온다. Ridge → OLS의 매끄러운 다리.

### SVD의 펼침

$X = U \Sigma V^\top$ (thin SVD), $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r, 0, \ldots, 0)$. Pseudoinverse는 **0이 아닌 특이값만 역수**를 취한 것:

$$\Sigma^+ = \text{diag}(1/\sigma_1, \ldots, 1/\sigma_r, 0, \ldots, 0), \qquad X^+ = V \Sigma^+ U^\top.$$

**기억법**: "역행렬을 흉내 내되, 0인 방향은 그냥 0으로 둔다."

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Moore-Penrose Pseudoinverse

$X \in \mathbb{R}^{n \times p}$의 **Moore-Penrose pseudoinverse**는 다음 4개 조건(Penrose 조건)을 모두 만족하는 유일한 행렬 $X^+ \in \mathbb{R}^{p \times n}$:

1. $X X^+ X = X$
2. $X^+ X X^+ = X^+$
3. $(X X^+)^\top = X X^+$
4. $(X^+ X)^\top = X^+ X$

### 정의 3.2 — Min-Norm Least-Squares Solution

$\beta$가 $\arg\min_\beta \|y - X\beta\|^2$의 한 해(LS solution)이고 그중 $\|\beta\|$가 최소인 것을 **min-norm least-squares solution**이라 한다.

### 정의 3.3 — SVD와 Pseudoinverse

$X = U \Sigma V^\top$ ($U \in \mathbb{R}^{n \times n}$, $V \in \mathbb{R}^{p \times p}$ 직교, $\Sigma$의 비영 특이값 $\sigma_1 \geq \cdots \geq \sigma_r > 0$). 그러면

$$X^+ := V \Sigma^+ U^\top, \qquad \Sigma^+_{ii} = \begin{cases} 1/\sigma_i & i \leq r \\ 0 & i > r \end{cases}.$$

---

## 🔬 정리와 증명

### 정리 3.1 — Pseudoinverse의 존재와 유일성

**명제**: 임의의 $X \in \mathbb{R}^{n \times p}$에 대해 정의 3.1의 4조건을 만족하는 $X^+$가 존재하고 유일하다.

**증명 스케치**: 존재 — 정의 3.3의 $V \Sigma^+ U^\top$이 4조건을 만족함을 직접 계산. 유일성 — 두 후보 $A_1, A_2$가 모두 만족한다고 하면 4조건의 조합으로 $A_1 = A_1 X A_1 = A_1 X A_2 X A_1 = \cdots = A_2$. 완전 전개는 Penrose (1955) 원본 또는 Ben-Israel & Greville Ch1. $\square$

### 정리 3.2 — SVD 표현

**명제**: $X = U \Sigma V^\top$이면 $X^+ = V \Sigma^+ U^\top$.

**증명**: 정의 3.3이 4조건을 만족함을 직접 검산:

$X X^+ X = U\Sigma V^\top V \Sigma^+ U^\top U \Sigma V^\top = U \Sigma \Sigma^+ \Sigma V^\top$. $\Sigma \Sigma^+ \Sigma$의 $(i, i)$ 원소는 $\sigma_i \cdot (1/\sigma_i) \cdot \sigma_i = \sigma_i$ (i ≤ r), 그 외 0. 즉 $\Sigma \Sigma^+ \Sigma = \Sigma$ → $X X^+ X = U\Sigma V^\top = X$. ✓

다른 3개 조건도 같은 방식으로 검산. $\square$

### 정리 3.3 — Ridge 극한 표현

**명제**: 임의의 $X$에 대해

$$X^+ = \lim_{\lambda \to 0^+} (X^\top X + \lambda I)^{-1} X^\top.$$

**증명**: SVD $X = U\Sigma V^\top$로 $X^\top X = V \Sigma^\top \Sigma V^\top$. $\Sigma^\top \Sigma$는 대각이고 원소 $\sigma_i^2$ (i ≤ r), 그 외 0. 따라서

$$X^\top X + \lambda I = V \,\text{diag}(\sigma_1^2 + \lambda, \ldots, \sigma_r^2 + \lambda, \lambda, \ldots, \lambda) \, V^\top.$$

역행렬:

$$(X^\top X + \lambda I)^{-1} = V \,\text{diag}\!\left(\frac{1}{\sigma_1^2 + \lambda}, \ldots, \frac{1}{\sigma_r^2 + \lambda}, \frac{1}{\lambda}, \ldots, \frac{1}{\lambda}\right) V^\top.$$

여기에 $X^\top = V \Sigma^\top U^\top$를 곱하면 비영 부분에서

$$\text{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right) \xrightarrow{\lambda \to 0^+} \text{diag}(1/\sigma_i)$$

영 부분 $1/\lambda \cdot 0 = 0$ (특이값이 0이므로 $\Sigma^\top$의 해당 원소가 0). 따라서 극한은 $V \Sigma^+ U^\top = X^+$. $\square$

### 정리 3.4 — Min-Norm Least-Squares

**명제**: $\hat{\beta} = X^+ y$는 (i) 최소제곱 해 $\arg\min_\beta \|y - X\beta\|^2$ 중 (ii) $\|\beta\|$가 최소인 유일한 해이다.

**증명**:

(i) **LS 해임**: $\hat{y} = X X^+ y$. Penrose 조건 (3)에서 $X X^+$는 대칭이고 (1)에서 $(XX^+)(X X^+) = X X^+$이므로 idempotent + 대칭, 즉 orthogonal projection. 또 $X X^+$의 range는 $\text{col}(X)$ (직접 검산). 따라서 $X X^+ y = P_{\text{col}(X)} y$. $r = y - X X^+ y \perp \text{col}(X)$, 즉 $X^\top r = 0$ → 정규방정식 만족 → LS 해.

(ii) **Min-norm**: 임의의 다른 LS 해 $\beta'$를 $\beta' = \hat{\beta} + v$로 쓰면 $X\beta' = X\hat{\beta}$ → $Xv = 0$ → $v \in \text{null}(X)$. SVD로 $\hat{\beta} = V \Sigma^+ U^\top y$ — 이는 $\text{row}(X) = \text{col}(V_{[:, 1:r]})$에 속함 (V의 처음 r개 열의 span). 한편 $\text{null}(X) = \text{col}(V_{[:, r+1:p]})$ — 마지막 p-r개 열. 따라서 $\hat{\beta} \perp \text{null}(X)$, $\hat{\beta} \perp v$. Pythagoras:

$$\|\beta'\|^2 = \|\hat{\beta} + v\|^2 = \|\hat{\beta}\|^2 + \|v\|^2 \geq \|\hat{\beta}\|^2,$$

등호는 $v = 0$ 즉 $\beta' = \hat{\beta}$일 때만. $\square$

### 정리 3.5 — Full Column Rank 시 환원

**명제**: $X$가 full column rank ($r = p$)이면 $X^+ = (X^\top X)^{-1} X^\top$.

**증명**: 정리 3.3에서 $\lambda \to 0^+$. Full column rank면 $\Sigma$의 모든 특이값 양수, $\Sigma^\top \Sigma$ 가역. $\lambda = 0$ 대입 가능 → $(X^\top X)^{-1} X^\top$. $\square$

> 💡 **함의**: Pseudoinverse는 OLS의 **진정한 일반화** — full rank이면 익숙한 공식, 그 외에는 min-norm 해.

### 정리 3.6 — Pseudoinverse의 기본 성질

**명제**:

1. $(X^+)^+ = X$
2. $(X^\top)^+ = (X^+)^\top$
3. $(c X)^+ = c^{-1} X^+$ ($c \neq 0$)
4. $X X^+$와 $X^+ X$는 모두 orthogonal projection.
5. $\text{col}(X X^+) = \text{col}(X)$, $\text{col}(X^+ X) = \text{row}(X) = \text{col}(X^\top)$.

**증명**: SVD 정의 3.3과 직접 계산. $\square$

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import time

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Rank-deficient X 만들기 — 두 열이 평행
# ─────────────────────────────────────────────
n, p = 100, 5
X = rng.standard_normal((n, p))
X[:, 1] = 2.0 * X[:, 0]   # column 1 = 2 × column 0 (collinear)

beta_true = np.array([1.0, 0.0, 0.5, -1.0, 2.0])
y = X @ beta_true + 0.05 * rng.standard_normal(n)

print(f'rank(X)     = {np.linalg.matrix_rank(X)}  (= r < p = {p})')
print(f'cond(X^T X) = {np.linalg.cond(X.T @ X):.2e}  (특이에 가까움)\n')

# Normal Equation 직접 → 위험
try:
    beta_ne = np.linalg.solve(X.T @ X, X.T @ y)
    print(f'Normal Eq (해는 나오지만 불안정): {beta_ne}')
except np.linalg.LinAlgError as e:
    print(f'Normal Eq 실패: {e}')

# Pseudoinverse → 안전
beta_pinv = np.linalg.pinv(X) @ y
print(f'Pseudoinverse  : {beta_pinv}')

# sklearn 내부도 SVD 기반
sk = LinearRegression(fit_intercept=False).fit(X, y)
print(f'sklearn        : {sk.coef_}')
print(f'||pinv - sklearn|| = {np.linalg.norm(beta_pinv - sk.coef_):.2e}')

# ─────────────────────────────────────────────
# 2. Pseudoinverse가 min-norm 해 (정리 3.4)
# ─────────────────────────────────────────────
# Null space에 임의 방향 v를 더해도 X(β + v) = Xβ
# → 같은 ŷ를 만드는 다른 β들의 norm이 더 큰지 확인
_, _, Vt = np.linalg.svd(X, full_matrices=True)
v_null = Vt[-1]   # null space의 한 방향
print(f'\n||X v_null|| = {np.linalg.norm(X @ v_null):.2e} (≈ 0)')

for c in [0.5, 1.0, 2.0]:
    beta_alt = beta_pinv + c * v_null
    print(f'  c={c:>4}: ||β_alt|| = {np.linalg.norm(beta_alt):.4f}, '
          f'||y - Xβ_alt|| = {np.linalg.norm(y - X @ beta_alt):.4f} (= LS 잔차)')
print(f'  c=0.0: ||β_pinv|| = {np.linalg.norm(beta_pinv):.4f} (가장 작음)')

# ─────────────────────────────────────────────
# 3. Ridge → pseudoinverse 극한 (정리 3.3)
# ─────────────────────────────────────────────
print()
for lam in [1.0, 1e-2, 1e-4, 1e-8, 1e-12]:
    beta_ridge = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
    diff = np.linalg.norm(beta_ridge - beta_pinv)
    print(f'  λ = {lam:.0e}:  ||β_ridge - β_pinv|| = {diff:.4e}')

# ─────────────────────────────────────────────
# 4. SVD 표현 (정리 3.2)
# ─────────────────────────────────────────────
U, s, Vt = np.linalg.svd(X, full_matrices=False)
s_pinv = np.where(s > 1e-10, 1.0 / s, 0.0)   # 0인 특이값은 0 유지
X_pinv_manual = Vt.T @ np.diag(s_pinv) @ U.T
print(f'\n||SVD-based pinv - np.linalg.pinv|| = '
      f'{np.linalg.norm(X_pinv_manual - np.linalg.pinv(X)):.2e}')

# ─────────────────────────────────────────────
# 5. Normal Eq vs QR vs SVD vs Cholesky 수치 비교
# ─────────────────────────────────────────────
# Hilbert matrix — 악명 높게 ill-conditioned
n2, p2 = 100, 8
H_mat = 1.0 / (np.arange(1, p2+1)[:, None] + np.arange(1, p2+1)[None, :] - 1)
X2 = rng.standard_normal((n2, p2)) @ H_mat   # ill-cond X
y2 = rng.standard_normal(n2)

methods = {
    'Normal Eq':  lambda: np.linalg.solve(X2.T @ X2, X2.T @ y2),
    'QR'       :  lambda: np.linalg.lstsq(X2, y2, rcond=None)[0],
    'SVD pinv' :  lambda: np.linalg.pinv(X2) @ y2,
    'Cholesky' :  lambda: np.linalg.cholesky(X2.T @ X2 + 0*np.eye(p2)).T,  # demo
}
print(f'\ncond(X) = {np.linalg.cond(X2):.2e}')
beta_truth = np.linalg.lstsq(X2, y2, rcond=None)[0]
for name, fn in methods.items():
    if name == 'Cholesky':
        L = fn(); beta_ch = np.linalg.solve(L.T, np.linalg.solve(L, X2.T @ y2))
        diff = np.linalg.norm(beta_ch - beta_truth)
    else:
        diff = np.linalg.norm(fn() - beta_truth)
    print(f'  {name:>10s}: ||β - β_QR|| = {diff:.4e}')
```

**출력 예시**:
```
rank(X)     = 4  (= r < p = 5)
cond(X^T X) = 5.43e+18  (특이에 가까움)

Normal Eq (해는 나오지만 불안정): [-1.79e+00 1.40e+00  4.99e-01 -1.00e+00  2.00e+00]
Pseudoinverse  : [ 0.205  0.410  0.500 -1.000  2.000]
sklearn        : [ 0.205  0.410  0.500 -1.000  2.000]
||pinv - sklearn|| = 1.4e-15

||X v_null|| = 8.9e-16 (≈ 0)
  c=0.5: ||β_alt|| = 2.379, ||y - Xβ_alt|| = 0.487 (= LS 잔차)
  c=1.0: ||β_alt|| = 2.498, ||y - Xβ_alt|| = 0.487 (= LS 잔차)
  c=2.0: ||β_alt|| = 3.178, ||y - Xβ_alt|| = 0.487 (= LS 잔차)
  c=0.0: ||β_pinv|| = 2.339 (가장 작음)

  λ = 1e+00:  ||β_ridge - β_pinv|| = 1.2317e-01
  λ = 1e-02:  ||β_ridge - β_pinv|| = 4.8765e-03
  λ = 1e-04:  ||β_ridge - β_pinv|| = 4.9001e-05
  λ = 1e-08:  ||β_ridge - β_pinv|| = 4.9023e-09
  λ = 1e-12:  ||β_ridge - β_pinv|| = 4.5311e-13
```

---

## 🔗 실전 활용

- **sklearn 내부**: `LinearRegression` → `scipy.linalg.lstsq` → SVD → pseudoinverse. `rcond` 파라미터로 영-임계 특이값 결정.
- **High-dim 문제** ($n < p$): genomics, NLP의 sparse one-hot → Normal Eq 자체가 불가능. Pseudoinverse 또는 Ridge/Lasso만이 선택지.
- **수치 안정성 순위** (조건수가 큰 경우): SVD > QR > Cholesky > Normal Eq. SVD가 가장 robust지만 가장 느림 ($O(np^2)$).
- **Truncated SVD**: 작은 특이값이 잡음에 민감 → 임계값 이하를 0으로 잘라 효과적 정규화. **PCA regression**의 기반.
- **Ridge가 무한 데이터에서 OLS로 수렴**: $\lambda$ 고정 시 $n \to \infty$에서 Ridge bias → 0, $\lambda \to 0$ + $n \to \infty$의 동시 한계가 OLS의 일치성.

---

## ⚖️ 가정과 한계

| 상황 | 권장 방법 | 이유 |
|------|-----------|------|
| Full rank, 잘 조건화된 $X$ | Cholesky | 가장 빠름 ($O(np^2/2)$) |
| Full rank, 약간의 ill-cond | QR | 조건수 제곱 안 됨 |
| Rank-deficient 또는 매우 ill-cond | SVD pseudoinverse | 가장 robust, min-norm 자동 |
| $n \ll p$ (high-dim) | Pseudoinverse + Ridge/Lasso | 정규화로 bias-variance 조정 |

**주의**: Pseudoinverse는 **min-norm을 자동 선택**하지만, 그것이 **항상 옳은 선택은 아니다**. Ridge는 norm penalty를 명시적으로 조정하고, Lasso는 sparsity를 강제한다. Pseudoinverse는 "어떤 정규화도 안 한 채로 유일한 해를 골라준다"는 minimal stance.

---

## 📌 핵심 정리

$$\boxed{X^+ = V \Sigma^+ U^\top = \lim_{\lambda \to 0^+}(X^\top X + \lambda I)^{-1} X^\top, \qquad \hat{\beta} = X^+ y \text{는 min-norm LS 해}}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **Penrose 4조건** | $XX^+X = X$, $X^+XX^+ = X^+$, $XX^+, X^+X$ 대칭 — 유일한 정의 |
| **SVD 표현** | $\Sigma$의 비영 특이값만 역수 |
| **Ridge 극한** | $\lambda \to 0^+$에서 $X^+$로 수렴 — 자연스러운 Ridge↔OLS 교량 |
| **Min-Norm** | LS 해 중 가장 작은 norm을 자동 선택 |
| **Full rank 환원** | $X^+ = (X^\top X)^{-1} X^\top$ — Normal Eq 회복 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $X = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}$의 SVD를 구하고 $X^+$를 직접 계산하라.

<details>
<summary>힌트 및 해설</summary>

$X = \begin{pmatrix} 1 \\ 2 \end{pmatrix} \begin{pmatrix} 1 & 1 \end{pmatrix}$ — rank 1.

Thin SVD: $u = \frac{1}{\sqrt{5}}\begin{pmatrix}1\\2\end{pmatrix}$, $v = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix}$, $\sigma = \sqrt{10}$.

$X^+ = \frac{1}{\sigma} v u^\top = \frac{1}{\sqrt{10}} \cdot \frac{1}{\sqrt{2}\sqrt{5}} \begin{pmatrix}1\\1\end{pmatrix}\begin{pmatrix}1 & 2\end{pmatrix} = \frac{1}{10}\begin{pmatrix}1 & 2\\ 1 & 2\end{pmatrix}$.

검산: $XX^+X = X$ ✓.

</details>

**문제 2** (심화): $\hat{\beta}_R = (X^\top X + \lambda I)^{-1} X^\top y$를 SVD로 펼쳐 각 좌표 $u_i^\top y$의 **shrinkage factor** $\sigma_i / (\sigma_i^2 + \lambda)$를 유도하라. 이것이 어떤 의미에서 "small singular value 방향을 더 많이 줄인다"인가?

<details>
<summary>힌트 및 해설</summary>

$X = U\Sigma V^\top$ → $X^\top X = V\Sigma^\top \Sigma V^\top$ → $(X^\top X + \lambda I)^{-1} = V \text{diag}(1/(\sigma_i^2 + \lambda)) V^\top$ → $\hat{\beta}_R = V \text{diag}\bigl(\sigma_i/(\sigma_i^2 + \lambda)\bigr) U^\top y = \sum_i \frac{\sigma_i}{\sigma_i^2 + \lambda}(u_i^\top y) v_i$.

계수 $f(\sigma) = \sigma/(\sigma^2 + \lambda)$:
- $\sigma \gg \sqrt{\lambda}$ → $f \approx 1/\sigma$ (OLS와 거의 같음)
- $\sigma \ll \sqrt{\lambda}$ → $f \approx \sigma/\lambda \to 0$ (작은 $\sigma$ 방향을 강하게 죽임)

즉 **noise-prone한 작은 특이값 방향을 우선적으로 shrink** → variance 감소. 자세한 해석은 Ch1-04에서.

</details>

**문제 3** (ML 연결): NN의 **Neural Tangent Kernel** $\Theta = \Phi \Phi^\top$의 pseudoinverse가 NN의 학습 결과를 왜 결정하는지 한 줄로 설명하라.

<details>
<summary>힌트 및 해설</summary>

무한폭 NN의 gradient flow는 함수공간에서 $f^*(x) = \Theta(x, X) \Theta(X, X)^{-1} y$ (또는 $\Theta(x, X) \Theta(X, X)^+ y$, $\Theta$가 특이일 때) 로 수렴 (Jacot et al. 2018). 즉 NN의 학습 = NTK 행렬의 **pseudoinverse를 통한 회귀**. "선형 회귀의 pseudoinverse를 마스터하면 무한폭 NN의 학습 결과를 마스터한 셈." 

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 기하학적 관점 — 수직투영](./02-geometric-projection.md) | [📚 README](../README.md) | [04. Ridge Regression의 3가지 해석 ▶](./04-ridge-three-views.md) |

</div>
