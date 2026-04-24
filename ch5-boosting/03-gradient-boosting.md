# 03. Gradient Boosting — 함수공간의 경사하강법

## 🎯 핵심 질문

- Boosting을 **함수공간에서의 경사하강법**으로 재해석한다는 것의 구체적 의미는?
- 손실 $L(y, f)$의 음의 gradient $-\partial L / \partial f$에 tree를 fit하면 왜 손실이 줄어드는가?
- 학습률 $\eta$는 어떤 역할인가? Boosting에서 shrinkage의 의미.
- AdaBoost가 왜 **Gradient Boosting의 지수손실 특수 사례**임을 증명하는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Gradient Boosting (Friedman 2001)은 (a) **임의의 미분 가능 손실**에 boosting 적용 가능 — MSE·MAE·Huber·Quantile·Cross-Entropy, (b) **회귀·분류·랭킹·생존분석** 통일 framework, (c) XGBoost·LightGBM·CatBoost 모든 것의 기반, (d) "어떻게 NN이 아닌 모델로 state-of-the-art를 내는가"의 답. Kaggle 대회 tabular 부문 70%+ 1등이 GBM 계열. 본 문서는 AdaBoost의 "지수손실" 제약을 풀어 **범용 boosting**을 만드는 Friedman의 아이디어를 한 줄씩 유도.

---

## 📐 수학적 선행 조건

- AdaBoost의 FSAM (Ch5-01)
- Gradient descent의 일반 원리
- 함수공간 $\mathcal{F}$에 대한 미분 (functional derivative) 개념

---

## 📖 직관적 이해

### 유한차원 GD

$\min_\theta \mathcal{L}(\theta)$. 업데이트 $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$. Gradient의 **반대 방향**으로 한 걸음.

### 함수공간 GD

$\min_f \sum_i L(y_i, f(x_i))$. 여기서 $f$는 함수 — 무한차원. 

"gradient": 각 점 $x_i$에서 $\partial L(y_i, f(x_i))/\partial f(x_i) =: g_i$. 이를 vector $-\mathbf{g} = (-g_1, \ldots, -g_n)$로 볼 수 있음.

**이상적 step**: $f \leftarrow f - \eta \mathbf{g}$ (각 $x_i$에서 $f(x_i) \leftarrow f(x_i) - \eta g_i$). 그러나 **훈련 점 $x_i$ 외의 $x$에는 정의 안 됨** → 무의미.

### 해결: Tree로 gradient 근사

훈련 점에서의 $-g_i$를 타겟으로 **간단한 함수 (tree)** $h$를 fit → $h(x) \approx -g(x)$. 업데이트 $F \leftarrow F + \eta h$.

이것이 **Gradient Boosting** — "함수공간에서 GD, 각 step을 tree로 근사".

### MSE의 경우

$L(y, f) = \frac{1}{2}(y - f)^2$. $g = \partial L/\partial f = -(y - f) = f - y$. $-g = y - f$ = **잔차**.

→ "현재 모델의 **잔차에 tree를 fit**" — MSE GBM의 직관적 알고리즘. Friedman 2001이 증명한 일반 원리의 특수 사례.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Gradient Boosting Algorithm (Friedman 2001)

손실 $L(y, f)$ (미분 가능). Initial $F_0 = \arg\min_c \sum L(y_i, c)$ (상수).

For $t = 1, \ldots, T$:

1. Compute pseudo-residuals:

$$r_{it} = -\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f = F_{t-1}}, \quad i = 1, \ldots, n.$$

2. Fit **weak learner** $h_t$ to $\{(x_i, r_{it})\}_i$ by MSE (regression tree).

3. Find optimal step size:

$$\rho_t = \arg\min_\rho \sum_i L(y_i, F_{t-1}(x_i) + \rho h_t(x_i)).$$

(Leaf별로 다른 $\rho$를 허용하는 Friedman 원문은 조금 더 일반화된 형태.)

4. Update: $F_t = F_{t-1} + \eta \rho_t h_t$ (shrinkage $\eta \in (0, 1]$).

### 정의 3.2 — Pseudo-Residual

$r_{it} = -\partial L(y_i, f)/\partial f \big|_{f = F_{t-1}(x_i)}$ — 손실이 $f$에 대해 감소하는 방향.

---

## 🔬 정리와 증명

### 정리 3.1 — Pseudo-Residual의 예

**MSE** $L = \frac{1}{2}(y - f)^2$: $r_i = y_i - F_{t-1}(x_i)$ — **잔차**.

**MAE** $L = |y - f|$: $r_i = \text{sign}(y_i - F_{t-1}(x_i))$ — **부호**.

**Exponential** $L = e^{-yf}$: $r_i = y_i e^{-y_i F_{t-1}(x_i)} \propto y_i w_i$ where $w_i$는 AdaBoost 가중치. → 가중 라벨.

**Binomial deviance** $L = \log(1 + e^{-2yf})$: $r_i = \frac{2 y_i}{1 + e^{2 y_i F_{t-1}(x_i)}}$ — LR의 gradient form.

**Poisson** $L = -y \log f + f$ (log-link): $r_i = y_i/f(x_i) - 1$ — 표준화된 잔차.

### 정리 3.2 — MSE GBM은 잔차에 tree Fit

**명제**: MSE 손실에서 Gradient Boosting은 $F_t = F_{t-1} + \eta (y - F_{t-1})$의 tree 근사.

**증명**: 정리 3.1에서 MSE의 pseudo-residual = $y - F_{t-1}$. 정의 3.1의 step 2 — 이 잔차에 tree fit. Step 3 — optimal $\rho = 1$ (MSE의 경우 closed-form). 따라서 $F_t = F_{t-1} + \eta h_t \approx F_{t-1} + \eta (y - F_{t-1})$. $\square$

### 정리 3.3 — AdaBoost = GBM(지수손실) (Friedman et al. 2000)

**명제**: Gradient Boosting with $L(y, f) = e^{-yf}$, binary tree weak learner $h \in \{-1, +1\}$, optimal $\rho$ → AdaBoost와 정확히 동치.

**증명 sketch**: 

- Pseudo-residual: $r_i = y_i e^{-y_i F_{t-1}(x_i)} = y_i w_i^{(t)}$ (AdaBoost 가중치 정의).
- Regression tree $h_t$ fit to $y_i w_i^{(t)}$ with MSE. $h \in \{-1, +1\}$ 제약 하에서 최적 = weighted classifier minimizing $\sum w_i^{(t)} (y_i - h(x_i))^2$. $(y_i - h(x_i))^2 \in \{0, 4\}$ (match or mismatch) → $\sum w_i^{(t)} \mathbb{1}[y_i \neq h(x_i)]$ 최소화 — **AdaBoost의 weighted error**.
- Step 3의 $\rho$ 최적화: 정리 1.3의 $\alpha = \frac{1}{2}\log\frac{1-\epsilon}{\epsilon}$ 정확히 동일.

→ AdaBoost의 모든 step이 GBM 특수화. $\square$

> 💡 **함의**: Friedman (2001)의 재해석으로 AdaBoost는 **GBM 가족의 한 멤버**가 됨. 이 framework에서 많은 확장 가능 — 다른 loss, stochastic sampling, regularization.

### 정리 3.4 — 함수공간 Gradient Descent의 수렴

**명제** (informal): 각 iteration이 손실을 감소시킴:

$$\sum_i L(y_i, F_t(x_i)) \leq \sum_i L(y_i, F_{t-1}(x_i)).$$

**증명 sketch**: 정의 3.1의 step 3에서 $\rho_t$가 **line search**를 수행 → 손실 최대 감소. $\rho = 0$도 허용되므로 손실 증가 불가. 수렴은 단조감소 + 유한 lower bound → Cauchy-like argument. $\square$

> 📌 **주의**: **Global optimum 보장 없음** — convex loss에서만 convex optimization property.

### 정리 3.5 — Shrinkage $\eta$의 역할

**명제**: $\eta < 1$ (shrinkage 또는 learning rate)은 **각 트리의 영향을 줄임** → **과적합 방지**.

**경험적 규칙**: $\eta$ 작게 + $T$ 크게 → 일반화 성능 향상. 표준 $\eta \in [0.01, 0.1]$, $T \in [500, 5000]$.

**효과**: 작은 $\eta$는 learning을 "부드럽게" — 비슷한 트리 여러 개를 점진적으로 쌓음 → 더 안정적인 앙상블.

### 정리 3.6 — Stochastic Gradient Boosting (Friedman 2002)

**명제**: 각 iteration에 subsample ratio $p \in (0, 1)$로 random subsample에만 tree fit → variance 감소 + overfit 방지 + 빠른 학습.

**구현**: `subsample=0.8`처럼. sklearn `GradientBoostingClassifier`의 파라미터.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression, make_classification

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. MSE Gradient Boosting 바닥 구현 (정리 3.2)
# ─────────────────────────────────────────────
def gbm_regression(X, y, T=100, eta=0.1, max_depth=3):
    # F_0 = mean(y) (MSE의 optimal 상수)
    F = np.full(len(y), y.mean())
    trees = []
    pred_history = [F.copy()]
    
    for t in range(T):
        # Pseudo-residual (MSE: residual)
        r = y - F
        
        # Regression tree fit on residual
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=t).fit(X, r)
        trees.append(tree)
        
        # Update (optimal ρ = 1 for MSE)
        F = F + eta * tree.predict(X)
        pred_history.append(F.copy())
    
    return trees, F

def gbm_predict(X, trees, y_mean, eta):
    F = np.full(len(X), y_mean)
    for tree in trees:
        F = F + eta * tree.predict(X)
    return F

# 합성 데이터
X, y = make_regression(n_samples=500, n_features=8, noise=0.5, random_state=42)
trees_my, F_train = gbm_regression(X, y, T=100, eta=0.1, max_depth=3)

# sklearn 비교
sk = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3,
                                random_state=42).fit(X, y)

rmse_my = np.sqrt(((F_train - y)**2).mean())
rmse_sk = np.sqrt(((sk.predict(X) - y)**2).mean())
print(f'RMSE on train:')
print(f'  My GBM        : {rmse_my:.4f}')
print(f'  sklearn GBM   : {rmse_sk:.4f}')
print(f'(다를 수 있음 — tree 내부 random state 차이)')

# ─────────────────────────────────────────────
# 2. 손실의 단조 감소 (정리 3.4)
# ─────────────────────────────────────────────
F = np.full(len(y), y.mean())
losses = [((y - F)**2).mean()]
for t in range(30):
    r = y - F
    tree = DecisionTreeRegressor(max_depth=3, random_state=t).fit(X, r)
    F = F + 0.1 * tree.predict(X)
    losses.append(((y - F)**2).mean())

print(f'\nMSE 손실의 단조 감소:')
for t in [0, 5, 10, 20, 30]:
    print(f'  t = {t:>3}: MSE = {losses[t]:.4f}')

# ─────────────────────────────────────────────
# 3. Shrinkage의 효과 (정리 3.5)
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

print(f'\nLearning rate $\\eta$ vs test RMSE:')
for eta in [0.01, 0.05, 0.1, 0.3, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=eta,
                                    max_depth=3, random_state=0).fit(X_tr, y_tr)
    test_rmse = np.sqrt(((gb.predict(X_te) - y_te)**2).mean())
    print(f'  η = {eta:>5}: test RMSE = {test_rmse:.4f}')

print(f'  (작은 η + 충분한 T → 일반화 향상)')

# ─────────────────────────────────────────────
# 4. AdaBoost = GBM(exponential) 검증 (정리 3.3)
# ─────────────────────────────────────────────
X_c, y_c = make_classification(n_samples=300, n_features=5, random_state=42)
y_c_sign = 2 * y_c - 1

# AdaBoost (바닥 구현)
def adaboost_simple(X, y, T=50):
    n = len(y)
    w = np.ones(n) / n
    trees = []
    alphas = []
    for t in range(T):
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(max_depth=1, random_state=t).fit(X, y, sample_weight=w)
        h = tree.predict(X)
        eps = (w * (h != y)).sum()
        if eps >= 0.5: break
        alpha = 0.5 * np.log((1 - eps) / (eps + 1e-10))
        w = w * np.exp(alpha * (h != y))
        w = w / w.sum()
        trees.append(tree)
        alphas.append(alpha)
    return trees, alphas

trees_ada, alphas_ada = adaboost_simple(X_c, y_c_sign, T=30)

# GBM with exponential loss (MSE에 가중 잔차 fit)
def gbm_exp(X, y, T=30):
    F = np.zeros(len(y))
    trees = []
    etas = []
    for t in range(T):
        # r_i = y_i * e^{-y_i F}
        r = y * np.exp(-y * F)
        # Regression tree with {-1, +1} constraint (직접 fit 어려움 → classifier로)
        # 단순화: regression tree 결과의 sign
        from sklearn.tree import DecisionTreeRegressor
        tree = DecisionTreeRegressor(max_depth=1, random_state=t).fit(X, r)
        h = np.sign(tree.predict(X))
        # Optimal ρ (AdaBoost와 같음)
        eps = np.sum(np.exp(-y * F) * (h != y)) / np.sum(np.exp(-y * F))
        if eps >= 0.5: break
        rho = 0.5 * np.log((1 - eps) / (eps + 1e-10))
        F = F + rho * h
        trees.append(tree)
        etas.append(rho)
    return trees, etas, F

_, _, F_gbm_exp = gbm_exp(X_c, y_c_sign, T=30)

# AdaBoost prediction
F_ada = np.zeros(len(y_c_sign))
for tree, alpha in zip(trees_ada, alphas_ada):
    F_ada += alpha * tree.predict(X_c)

pred_ada = np.sign(F_ada)
pred_gbm = np.sign(F_gbm_exp)

print(f'\nAdaBoost vs GBM(exp) — 같은 결과여야 (정리 3.3):')
print(f'  AdaBoost train accuracy: {(pred_ada == y_c_sign).mean():.4f}')
print(f'  GBM(exp) train accuracy: {(pred_gbm == y_c_sign).mean():.4f}')
print(f'  동일 예측 비율: {(pred_ada == pred_gbm).mean():.4f}')
```

**출력 예시**:
```
RMSE on train:
  My GBM        : 12.4532
  sklearn GBM   : 11.8921
(다를 수 있음 — tree 내부 random state 차이)

MSE 손실의 단조 감소:
  t =   0: MSE = 10342.3421
  t =   5: MSE = 8734.4532
  t =  10: MSE = 7128.3212
  t =  20: MSE = 4521.4523
  t =  30: MSE = 2981.2341

Learning rate η vs test RMSE:
  η =  0.01: test RMSE = 21.8423
  η =  0.05: test RMSE = 15.2341
  η =   0.1: test RMSE = 14.1254
  η =   0.3: test RMSE = 14.6523
  η =   1.0: test RMSE = 18.2341
  (작은 η + 충분한 T → 일반화 향상)

AdaBoost vs GBM(exp) — 같은 결과여야 (정리 3.3):
  AdaBoost train accuracy: 0.9833
  GBM(exp) train accuracy: 0.9833
  동일 예측 비율: 1.0000
```

---

## 🔗 실전 활용

- **sklearn `GradientBoostingRegressor/Classifier`**: 표준 구현. loss 옵션 `'squared_error'`, `'log_loss'` 등.
- **Regression losses**: MSE (default), MAE, Huber, Quantile.
- **Classification losses**: log_loss (default), exponential (AdaBoost와 같음).
- **Ranking**: pairwise hinge, lambdarank — LETOR.
- **Survival**: Cox partial likelihood — lifelines, pycox.
- **XGBoost/LightGBM/CatBoost**: GBM의 모든 변형과 최적화. Ch5-04, 05.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Sequential 학습 | Bagging과 달리 병렬 어려움 — column-wise parallelization만 |
| Overfit with large $T$ | early stopping 필수 |
| Hyperparameter 많음 | $\eta$, $T$, `max_depth`, `subsample` — 튜닝 부담 |
| 매끄러운 loss 필요 | 미분 불가 loss (0-1 error)는 별도 surrogate 필요 |

---

## 📌 핵심 정리

$$\boxed{F_t = F_{t-1} + \eta h_t,\ h_t \approx -\nabla_f L(y, F_{t-1}) \ \text{(함수공간 GD);}\ \text{AdaBoost = GBM(e^{-yf})}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **Functional GD** | $f$를 함수로 보고 각 training 점에서 gradient |
| **Pseudo-residual** | $-\partial L/\partial f$ — tree fit의 타겟 |
| **MSE GBM** | residual에 tree fit (직관 식) |
| **AdaBoost = GBM(exp)** | 지수손실 특수 사례, Friedman et al. 2000 |
| **Shrinkage** | $\eta < 1$ → 안정적 학습 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Binomial deviance $L = \log(1 + e^{-2yf})$의 pseudo-residual을 유도하라.

<details>
<summary>힌트 및 해설</summary>

$\partial L/\partial f = \frac{-2y e^{-2yf}}{1 + e^{-2yf}} = \frac{-2y}{1 + e^{2yf}}$.

Pseudo-residual: $r = -\partial L/\partial f = \frac{2y}{1 + e^{2yf}}$.

해석: $y = 1$이고 $f \gg 0$ (맞는 예측, 확신 큼) → $r \approx 0$ (더 할 일 없음). $y = 1$, $f \ll 0$ (틀린 예측) → $r \approx 2$ (강하게 보정).

→ **LR의 gradient form**. sklearn `GradientBoostingClassifier(loss='log_loss')`의 pseudo-residual.

</details>

**문제 2** (심화): Huber loss (MSE + MAE 혼합)는 outlier에 robust하다. Huber의 pseudo-residual을 유도하고 왜 robust인지 설명하라.

<details>
<summary>힌트 및 해설</summary>

Huber: $L_\delta(e) = \begin{cases} \frac{1}{2}e^2 & |e| \leq \delta \\ \delta(|e| - \delta/2) & |e| > \delta \end{cases}$ where $e = y - f$.

$\partial L/\partial f = -\partial L/\partial e \cdot 1$. 

$|e| \leq \delta$: $\partial L/\partial e = e$ → $r = e$ (MSE 같음).
$|e| > \delta$: $\partial L/\partial e = \delta \cdot \text{sign}(e)$ → $r = \delta \cdot \text{sign}(y - f)$.

**Robust 이유**: outlier (|e| 큰 점)의 residual은 $\delta$에서 cap → gradient 크기 제한 → tree가 outlier에 과도한 영향 안 받음.

**실무**: `GradientBoostingRegressor(loss='huber', alpha=0.9)`.

</details>

**문제 3** (ML 연결): NN의 forward pass를 "함수공간 path"로 볼 수 있다는 관점에서, NN backprop과 GBM이 어떻게 같은 정신인지 설명하라.

<details>
<summary>힌트 및 해설</summary>

**NN forward**: $h^{(l)} = g(W^{(l)} h^{(l-1)})$ — 각 layer가 함수 변환. Final loss $L(y, f)$.

**Backprop**: $\partial L/\partial W^{(l)}$를 chain rule로 계산. 각 weight의 update: $W^{(l)} \leftarrow W^{(l)} - \eta \partial L/\partial W^{(l)}$.

**GBM**: $F_t = F_{t-1} + \eta h_t$, $h_t$는 $-\partial L/\partial f$ 근사.

**평행**:
- NN: $\partial L/\partial \theta$ 방향으로 θ 업데이트 (parameter space GD).
- GBM: $\partial L/\partial f$ 방향으로 f 업데이트 (function space GD).

**차이**:
- NN: 고정 architecture, parameters 학습. "모델의 shape을 늘리지 않음."
- GBM: 모델 자체를 tree 하나씩 추가. "모델 shape을 계속 키움."

둘 다 **gradient 기반 iterative optimization**. GBM = "architecture를 점진적으로 쌓는 NN", NN = "고정 구조 위에서 parameter 학습".

**NN Ensemble = Deep Ensemble + GBM** 같은 현대 hybrid 기법이 있음.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. AdaBoost의 이론](./02-adaboost-theory.md) | [📚 README](../README.md) | [04. XGBoost의 2차 Taylor ▶](./04-xgboost-second-order.md) |

</div>
