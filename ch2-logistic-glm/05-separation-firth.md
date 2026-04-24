# 05. 분리 문제(Separation Problem)와 Firth Correction

## 🎯 핵심 질문

- 두 클래스가 **완전 분리(complete separation)** 가능하면 왜 LR의 MLE가 **무한 발산**하는가?
- 발산의 **기하학적 직관** — 가능한 경계를 무한대로 옮기는 한 어떤 점에서도 더 큰 likelihood가 나오는 이유는?
- **Firth's penalized likelihood** $\ell(\beta) + \frac{1}{2}\log|I(\beta)|$가 어떻게 유한해를 보장하는가?
- Bayesian 관점 — Firth 보정이 **Jeffreys prior** MAP과 같다는 것의 의미는?

---

## 🔍 왜 이 개념이 ML에서 중요한가

분리 문제는 (a) **희귀 사건 분류** (rare events: 1만 명 중 5명만 사기 거래) 등 imbalanced data에서 매우 흔함, (b) **high-dim regime** ($p \to n$)에서 거의 보장됨, (c) MLE의 표준 가정인 "interior solution"이 깨지는 대표 사례, (d) 해결책 (Ridge, Firth, Bayesian prior)이 **모두 prior/penalty를 추가하는 같은 원리** — 즉 분리 문제를 통해 **"왜 정규화가 필수인가"를 가장 극적인 사례로** 본다. 본 문서는 sklearn 기본값 `C = 1.0`이 사실 **분리 보호 장치**임을 알려준다.

---

## 📐 수학적 선행 조건

- LR의 MLE와 Hessian의 PSD성 (Ch2-01)
- Ridge regression과 정규화 (Ch1-04)
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): Fisher information, Jeffreys prior

---

## 📖 직관적 이해

### Separation의 정의

**Complete separation**: 어떤 hyperplane $w^\top x + b = 0$이 모든 데이터 점을 완벽히 분리 — $w^\top x_i + b > 0$ for $y_i = 1$, $w^\top x_i + b < 0$ for $y_i = 0$.

**Quasi-complete separation**: 위와 같지만 $\geq 0$, $\leq 0$ (경계 위에 점 허용).

분리되면 **임의의 $c > 0$에 대해** $cw, cb$로 weight를 키우면 모든 $p_i \to 1$ ($y = 1$인 점) 또는 $\to 0$ ($y = 0$인 점) → likelihood $\to 1$ → log-likelihood $\to 0$ (위로부터 sup). 그러나 어디서도 도달 안 됨 → **MLE 비존재**.

### Hessian 무너짐

$p_i \to 0$ or $1$이면 $W_{ii} = p_i(1-p_i) \to 0$ → Hessian $-X^\top W X \to 0$ → 거의 평평해짐 → Newton update 무한히 큼.

### Firth 보정의 아이디어

$\ell(\beta)$가 발산하는 방향에서 **Fisher information determinant** $|I(\beta)| = |X^\top W X|$도 **0으로 수렴** ($W \to 0$이므로 $|X^\top W X| \to 0$). 따라서 $\frac{1}{2}\log|I(\beta)| \to -\infty$ — **분리 방향에서 페널티가 무한대로 커짐** → 발산 방지.

$$\ell^*(\beta) := \ell(\beta) + \frac{1}{2}\log|I(\beta)|.$$

이 페널티는 **데이터 의존적** ($X$ 구조에 자동으로 적응) — 단순한 $\lambda \|\beta\|^2$ 보다 더 정교.

### Bayesian Jeffreys Prior

Jeffreys prior: $\pi_J(\beta) \propto |I(\beta)|^{1/2}$. 그 MAP은 $\arg\max_\beta \log p(\beta \mid y) = \arg\max [\ell(\beta) + \log \pi_J(\beta)] = \arg\max [\ell(\beta) + \frac{1}{2}\log |I(\beta)|]$ — **Firth와 정확히 일치**.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Complete Separation

데이터 $\{(x_i, y_i)\}$가 **complete separation**된다는 것은 어떤 $\beta \in \mathbb{R}^p$가 존재해 

$$y_i = 1 \implies x_i^\top \beta > 0, \quad y_i = 0 \implies x_i^\top \beta < 0$$

이 모든 $i$에서 성립하는 것이다.

### 정의 5.2 — Firth's Penalized Likelihood

$$\ell^*(\beta) := \ell(\beta) + \frac{1}{2}\log |I(\beta)| = \ell(\beta) + \frac{1}{2}\log |X^\top W X|.$$

Firth MLE: $\hat{\beta}_F := \arg\max \ell^*(\beta)$.

### 정의 5.3 — Jeffreys Prior

$$\pi_J(\beta) \propto |I(\beta)|^{1/2} = |X^\top W X|^{1/2}.$$

이는 reparameterization-invariant prior. (좌표 변환에 대해 분포가 보존되는 유일한 noninformative prior 종류.)

---

## 🔬 정리와 증명

### 정리 5.1 — 분리 시 MLE 비존재

**명제**: 데이터가 complete separation되면 $\sup_\beta \ell(\beta) = 0$이지만 어떤 유한 $\beta$에서도 도달하지 않음.

**증명**: separation을 구현하는 $\beta_0$를 잡고 $\beta_t := t \beta_0$ 살펴봄. $t \to \infty$이면 $x_i^\top \beta_t \to +\infty$ ($y_i = 1$) 또는 $-\infty$ ($y_i = 0$) → $p_i \to 1$ or $0$ correctly. 

$\ell(\beta_t) = \sum [y_i \log p_i + (1-y_i)\log(1-p_i)] \to 0$ (각 항 $\to 0$).

그러나 어떤 유한 $t$에서도 $p_i$가 정확히 1/0이 안 되므로 $\ell < 0$ — sup 미달성. $\square$

> 💡 **수치적으로**: Newton 업데이트가 발산. 각 step에서 $\|\beta\|$가 무한히 커짐. sklearn에서는 `max_iter` 도달로 멈춤 (warning).

### 정리 5.2 — Firth 보정 하 MLE 존재

**명제**: 데이터가 complete separation되어도 $\ell^*(\beta) = \ell(\beta) + \frac{1}{2}\log|X^\top W X|$의 최대값이 유한 $\beta$에서 달성된다 (under mild conditions).

**증명 스케치**: $\beta \to \infty$ 방향 (분리)에서 $W \to 0$ → $|X^\top W X| \to 0$ → $\log|X^\top W X| \to -\infty$. 따라서 $\ell^*(\beta) \to -\infty$. 한편 origin 근처에서 $\ell^* > -\infty$. 컴팩트 sublevel set + 연속 → 유한 max. $\square$

> 📌 **핵심**: 발산 방향에서 **두 가지 경쟁** — likelihood는 $0$로 올라가지만 penalty가 더 빠르게 $-\infty$로 떨어짐. 균형점이 유한.

### 정리 5.3 — Firth = Jeffreys MAP

**명제**: Jeffreys prior $\pi_J(\beta) \propto |I(\beta)|^{1/2}$를 사용한 MAP는 Firth 추정량과 일치.

**증명**: $\log p(\beta \mid y) = \ell(\beta) + \log \pi_J(\beta) - \log p(y)$. $\log \pi_J = \frac{1}{2}\log |I(\beta)| + \text{const}$. MAP = $\arg\max [\ell + \frac{1}{2}\log|I|]$ — Firth. $\square$

> 💡 **Bayesian 정당화**: Firth는 ad hoc 페널티가 아닌 **자연스러운 noninformative prior의 MAP**. Reparameterization에 invariant — 어떤 변수 변환에서도 같은 답을 줌.

### 정리 5.4 — Firth의 점근적 성질

**명제**: $n$이 충분히 크고 separation이 없으면 Firth MLE는 일반 MLE와 같은 점근분포 — $\sqrt{n}(\hat{\beta}_F - \beta) \xrightarrow{d} \mathcal{N}(0, I^{-1}/n)$. 또한 **$O(1/n)$ 정도의 bias 보정** — 일반 MLE의 first-order bias를 정확히 제거 (Firth 1993).

**증명 스케치**: Firth penalty가 $O(1)$인 반면 likelihood가 $O(n)$ — 큰 $n$에서는 영향이 점근적으로 사라짐. 정확한 bias 항 분석은 cumulant expansion (Firth 1993). $\square$

### 정리 5.5 — Ridge vs Firth 비교

| 측면 | L2 Ridge | Firth |
|------|----------|-------|
| 페널티 | $\frac{\lambda}{2}\|\beta\|^2$ | $\frac{1}{2}\log|X^\top W X|$ |
| Tuning | $\lambda$ 필요 (CV) | tuning-free |
| Coordinate-invariant? | ❌ (변수 scaling 의존) | ✅ |
| Bias 보정 | shrinkage 정보로 trading | first-order bias 제거 |
| 계산 | 매우 빠름 | $|X^\top W X|$ 계산 필요 ($O(p^3)$) |
| 분리 보호 | ✅ ($\lambda > 0$이면) | ✅ |

**실무 권장**: 빠른 baseline은 Ridge, **드문 사건 (rare events)**의 estimator 정확성 중요하면 Firth.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(42)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ─────────────────────────────────────────────
# 1. Separable한 데이터 만들기 — MLE 발산 확인
# ─────────────────────────────────────────────
n = 50
X = rng.standard_normal((n, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(int)   # 완벽 separable
X = np.hstack([np.ones((n, 1)), X])
print(f'데이터 구조: y = 1 if x_1 + x_2 > 0 — separable')
print(f'  y=0 개수: {(y == 0).sum()}, y=1 개수: {(y == 1).sum()}')

# Plain Newton-Raphson (정규화 없음)
def newton(X, y, n_iter=100, tol=1e-10, ridge=0.0):
    p = X.shape[1]
    beta = np.zeros(p)
    history = []
    for it in range(n_iter):
        prob = sigmoid(X @ beta)
        prob = np.clip(prob, 1e-12, 1 - 1e-12)
        g = X.T @ (y - prob) - ridge * beta
        W = prob * (1 - prob)
        H = X.T @ (W[:, None] * X) + ridge * np.eye(p)
        try:
            update = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta_new = beta + update
        history.append(np.linalg.norm(beta_new))
        if np.linalg.norm(update) < tol:
            break
        beta = beta_new
    return beta, history, it + 1

beta_no_reg, hist, n_iter = newton(X, y, n_iter=50, ridge=0.0)
print(f'\n정규화 없는 Newton:')
print(f'  iter 수: {n_iter} (max=50, 도달했으면 발산 중)')
print(f'  최종 ||β|| = {np.linalg.norm(beta_no_reg):.4f}')
print(f'  ||β|| 변화: {[f"{h:.1f}" for h in hist[::5]][:5]}')

# Ridge 적용 — 안정화
beta_ridge, _, _ = newton(X, y, ridge=1.0)
print(f'\nRidge (λ=1.0):')
print(f'  β = {beta_ridge.round(3)}')
print(f'  ||β|| = {np.linalg.norm(beta_ridge):.4f}')

# sklearn 기본 (C=1.0이 사실 ridge)
sk = LogisticRegression(fit_intercept=False, C=1.0).fit(X, y)
print(f'\nsklearn (C=1.0, 즉 λ=1):')
print(f'  β = {sk.coef_[0].round(3)}')
print(f'  → 발산 안 함 (separation 보호됨)')

# C=1e10 (무 regularization 흉내)
sk_big = LogisticRegression(fit_intercept=False, C=1e10, max_iter=100).fit(X, y)
print(f'\nsklearn (C=1e10, regularization 거의 없음):')
print(f'  ||β|| = {np.linalg.norm(sk_big.coef_):.4f} (큼 — 발산 시작)')

# ─────────────────────────────────────────────
# 2. Firth 추정량 직접 구현
# ─────────────────────────────────────────────
def firth_logit(X, y, n_iter=100, tol=1e-8):
    """Firth's penalized likelihood logistic regression"""
    n, p = X.shape
    beta = np.zeros(p)
    for it in range(n_iter):
        prob = sigmoid(X @ beta)
        prob = np.clip(prob, 1e-12, 1 - 1e-12)
        W = prob * (1 - prob)
        I_mat = X.T @ (W[:, None] * X)   # Fisher info
        # Firth의 hat matrix: H = X (X^T W X)^-1 X^T W
        try:
            H_diag = np.einsum('ij,jk,ik->i', X * W[:, None], 
                                np.linalg.inv(I_mat), X)
        except np.linalg.LinAlgError:
            break
        # Modified score: U* = X^T (y - p + h*(0.5 - p))
        adjustment = H_diag * (0.5 - prob)
        score = X.T @ (y - prob + adjustment)
        try:
            update = np.linalg.solve(I_mat, score)
        except np.linalg.LinAlgError:
            break
        beta_new = beta + update
        if np.linalg.norm(update) < tol:
            break
        beta = beta_new
    return beta, it + 1

beta_firth, firth_iter = firth_logit(X, y)
print(f'\nFirth 추정량:')
print(f'  β = {beta_firth.round(3)}')
print(f'  ||β|| = {np.linalg.norm(beta_firth):.4f} (유한)')
print(f'  iter 수: {firth_iter}')

# ─────────────────────────────────────────────
# 3. Non-separable 데이터 — Firth와 MLE가 가까움
# ─────────────────────────────────────────────
n = 500
X2 = rng.standard_normal((n, 2))
prob2 = sigmoid(0.5 * X2[:, 0] - 0.3 * X2[:, 1])
y2 = (rng.uniform(size=n) < prob2).astype(int)
X2 = np.hstack([np.ones((n, 1)), X2])

beta_mle2, _, _ = newton(X2, y2, ridge=0.0)
beta_firth2, _ = firth_logit(X2, y2)
print(f'\nNon-separable, n={n}:')
print(f'  MLE  β = {beta_mle2.round(4)}')
print(f'  Firth β = {beta_firth2.round(4)}')
print(f'  차이 ||MLE - Firth|| = {np.linalg.norm(beta_mle2 - beta_firth2):.4f}')
print(f'  (큰 n에서 점근적으로 같음 — 정리 5.4)')
```

**출력 예시**:
```
데이터 구조: y = 1 if x_1 + x_2 > 0 — separable
  y=0 개수: 26, y=1 개수: 24

정규화 없는 Newton:
  iter 수: 50 (max=50, 도달했으면 발산 중)
  최종 ||β|| = 215.34
  ||β|| 변화: ['1.5', '14.1', '49.8', '154.2']
  ...

Ridge (λ=1.0):
  β = [-0.169  1.142  1.103]
  ||β|| = 1.601

sklearn (C=1.0, 즉 λ=1):
  β = [-0.169  1.143  1.103]
  → 발산 안 함 (separation 보호됨)

sklearn (C=1e10, regularization 거의 없음):
  ||β|| = 1245.32 (큼 — 발산 시작)

Firth 추정량:
  β = [-0.045  0.821  0.795]
  ||β|| = 1.143 (유한)
  iter 수: 7

Non-separable, n=500:
  MLE  β = [ 0.012  0.5128 -0.3041]
  Firth β = [ 0.011  0.5102 -0.3026]
  차이 ||MLE - Firth|| = 0.0035
  (큰 n에서 점근적으로 같음 — 정리 5.4)
```

---

## 🔗 실전 활용

- **Rare events** (의료·사기 탐지, 5% 미만의 minority class): Firth가 표준. R `logistf` 패키지.
- **High-dim regime** ($p \approx n$): separation 거의 보장 → Ridge 또는 Firth 필수.
- **Imbalanced classification**: Firth가 small class의 estimate를 stabilize.
- **Profile likelihood CI**: Firth는 좁은 신뢰구간을 줌 — Wald CI보다 정확.
- **Python**: `firthlogist` 패키지 또는 statsmodels에 직접 구현 가능.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Firth가 estimator를 바꿈 | 점근적으로 같지만 small $n$에서 차이 — bias 보정이지만 다른 inference |
| Firth가 약간 conservative | shrinkage가 약간 강함 — Ridge가 같은 정도 |
| 계산 비용 | $|X^\top W X|$ 계산 — 매 iteration $O(p^3)$ |
| Multinomial Firth | 표준 정의가 더 복잡 — Bull et al. (2007) |

---

## 📌 핵심 정리

$$\boxed{\text{Separation 시 } \sup \ell = 0 \text{ 미달성, MLE 비존재.\ Firth: } \ell^* = \ell + \tfrac{1}{2}\log|X^\top W X| = \text{Jeffreys MAP}}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **Complete separation** | hyperplane이 완벽 분리 → MLE 발산 |
| **수치 증상** | $\|\beta\|$ 무한 증가, $W \to 0$, Hessian → 0 |
| **Firth 보정** | 데이터 의존 페널티로 발산 방향에서 $-\infty$ 추가 |
| **Jeffreys MAP** | Firth = $\pi_J(\beta) \propto |I(\beta)|^{1/2}$ MAP |
| **Ridge 대안** | $\lambda \|\beta\|^2$ — 더 빠르지만 less principled |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 단변수 ($p = 1$, no intercept) LR에서 $y = (x > 0)$이면 separation. $\ell(\beta) = ?$ ($\beta \to \infty$로). 발산함을 직접 보여라.

<details>
<summary>힌트 및 해설</summary>

$y_i = 1$ if $x_i > 0$, else $0$. $\ell(\beta) = \sum_{x_i > 0} \log \sigma(\beta x_i) + \sum_{x_i < 0} \log \sigma(-\beta x_i) = \sum \log \sigma(|\beta x_i|)$ (모든 항이 같은 부호).

$\beta \to \infty$이면 $|\beta x_i| \to \infty$, $\sigma \to 1$, $\log \to 0$ → $\ell \to 0^-$. 어떤 유한 $\beta$에서도 미달성. 발산.

</details>

**문제 2** (심화): Ridge LR과 Firth 모두 separation을 해결한다. 둘의 **bias 행동**이 다른 이유는?

<details>
<summary>힌트 및 해설</summary>

**Ridge**: $\lambda \|\beta\|^2$가 모든 좌표를 0 쪽으로 동등하게 shrink — $\beta$ 추정값에 **systematic bias** ($\hat{\beta}_R \to 0$ as $\lambda \to \infty$). 이 bias는 $n \to \infty$에서도 (만약 $\lambda$ 고정) 사라지지 않음.

**Firth**: penalty가 $O(1)$, $\ell$이 $O(n)$ — 큰 $n$에서 영향 사라짐. 게다가 **$O(1/n)$의 first-order bias를 정확히 제거** (Firth 1993). 즉 점근적으로 unbiased + small-sample bias 보정.

**결론**: rare events에서 estimator의 정확성 중요 → Firth. 빠른 baseline 또는 sparse → Ridge.

</details>

**문제 3** (ML 연결): NN의 분류 head에서 separation은 어떻게 발생할 수 있는가? Modern NN이 항상 weight decay를 쓰는 이유와 연결지어라.

<details>
<summary>힌트 및 해설</summary>

NN의 hidden representation $\phi(x) \in \mathbb{R}^d$가 충분히 깊고 expressive하면, 분류 train set에서 두 클래스가 **항상 separable**가 됨 (over-parameterized regime). 그러면 마지막 layer의 weight $w$는 무한 발산 — `||w||` 폭발.

**Weight decay** (= L2 regularization): 매 step gradient에 $-\lambda w$를 더해 $w$를 0 쪽으로 pull → 분리 방향 폭발 방지. 즉 modern NN의 weight decay는 사실 **logit separation에 대한 Ridge LR 보호 장치**.

**대안**: BatchNorm/LayerNorm은 activation의 scale을 통제 → 간접적으로 logit scale 제한 → separation 효과 완화. **Label smoothing**은 one-hot을 smooth하게 → 정확히 1/0 도달 불가 → MLE 발산 자체를 정의 불가능하게.

**결론**: Modern NN의 (weight decay + BN + label smoothing) 셋트는 모두 LR의 separation 문제를 다른 layer에서 해결하는 도구.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Multinomial·Softmax](./04-softmax-multinomial.md) | [📚 README](../README.md) | [Ch3-01. Information Gain ▶](../ch3-decision-tree/01-information-gain.md) |

</div>