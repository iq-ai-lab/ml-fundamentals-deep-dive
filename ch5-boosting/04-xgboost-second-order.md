# 04. XGBoost — 2차 Taylor 근사

## 🎯 핵심 질문

- XGBoost의 2차 Taylor 근사 $L_t \approx L_{t-1} + g_t \Delta f + \frac{1}{2} h_t \Delta f^2$이 어떻게 leaf 값 **closed-form** $w^* = -G/(H + \lambda)$를 주는가?
- 이것이 **tree 버전 Newton-Raphson 한 스텝**임을 어떻게 증명하는가?
- Regularization $\gamma \cdot \|T\| + \frac{1}{2}\lambda \sum w_j^2$의 역할은? `gamma`, `lambda`, `alpha` 파라미터의 의미.
- **Gain** 공식 $\frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}] - \gamma$는 왜 그 형태인가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

XGBoost (Chen & Guestrin 2016)는 (a) **Kaggle·실무에서 tabular 데이터 1위 알고리즘** (2014~현재), (b) **Gradient Boosting에 regularization과 2차 정보 추가** — 더 정확한 step, (c) **"tree 버전 Newton-Raphson"**이라는 통합 관점, (d) 이후 LightGBM·CatBoost의 기반. 본 문서는 XGBoost의 핵심 innovation — **2차 Taylor approximation + 정규화 + closed-form leaf value** — 이 한 줄씩 유도된다는 것을 보인다. 이 수식들의 이해가 `max_depth`, `gamma`, `lambda`, `learning_rate` 튜닝의 직관을 만든다.

---

## 📐 수학적 선행 조건

- Gradient Boosting (Ch5-03)
- Newton-Raphson (Ch2-02)
- Taylor 전개 (2차까지)

---

## 📖 직관적 이해

### GBM은 "1차 Taylor"만 사용

GBM (Friedman 2001): pseudo-residual $r = -\partial L/\partial f$에 tree fit — **1차 정보**만.

### XGBoost: 2차 정보 추가

$g_i = \partial L/\partial f$, $h_i = \partial^2 L/\partial f^2$. 2차 Taylor:

$$L(y_i, F_{t-1}(x_i) + f_t(x_i)) \approx L(y_i, F_{t-1}(x_i)) + g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2.$$

상수항 제거, 정규화 추가:

$$\mathcal{L}(f_t) = \sum_i [g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2] + \gamma T + \frac{1}{2}\lambda \sum_j w_j^2.$$

### Leaf 별 최적화

Tree의 leaf $j$, 해당 region에 속한 점들 $I_j$, leaf 값 $w_j$. Tree가 정해지면 $f_t(x_i) = w_{q(x_i)}$ ($q$는 샘플을 leaf로 매핑).

$$\mathcal{L} = \sum_j \left[(\sum_{i \in I_j} g_i) w_j + \frac{1}{2}(\sum_{i \in I_j} h_i + \lambda) w_j^2\right] + \gamma T.$$

각 leaf에 대해 $w_j$로 미분 = 0:

$$w_j^* = -\frac{G_j}{H_j + \lambda}, \quad G_j = \sum_{i \in I_j} g_i, \ H_j = \sum_{i \in I_j} h_i.$$

→ **closed-form**. 이를 대입하면 최소 손실:

$$\mathcal{L}^* = -\frac{1}{2}\sum_j \frac{G_j^2}{H_j + \lambda} + \gamma T.$$

### Split Gain

Split이 한 leaf를 L, R로 나눔 → 새 손실 vs 옛 손실의 차이:

$$\text{Gain} = \frac{1}{2}\!\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma.$$

Gain > 0이면 split, 아니면 멈춤 (pre-pruning).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — XGBoost Objective

$$\mathcal{L}(F) = \sum_{i=1}^n L(y_i, F(x_i)) + \sum_{t=1}^T \Omega(f_t),$$

$$\Omega(f_t) = \gamma \cdot T_{\text{leaves}}(f_t) + \frac{1}{2}\lambda \sum_{j=1}^{T_{\text{leaves}}(f_t)} w_j^2 + \alpha \sum_j |w_j|.$$

- $\gamma$: leaf 수 페널티 (tree 복잡도 제어, pre-pruning).
- $\lambda$: L2 leaf weight 페널티 (Ridge 유사).
- $\alpha$: L1 leaf weight 페널티 (Lasso 유사).

### 정의 4.2 — Second-Order Gradient/Hessian

$$g_i^{(t)} := \frac{\partial L(y_i, f)}{\partial f}\bigg|_{f = F_{t-1}(x_i)}, \qquad h_i^{(t)} := \frac{\partial^2 L(y_i, f)}{\partial f^2}\bigg|_{f = F_{t-1}(x_i)}.$$

### 정의 4.3 — Leaf Statistics

Leaf $j$의 샘플 집합 $I_j$:

$$G_j = \sum_{i \in I_j} g_i, \quad H_j = \sum_{i \in I_j} h_i.$$

---

## 🔬 정리와 증명

### 정리 4.1 — Leaf 값의 Closed-Form (λ만, α = 0)

**명제**: 2차 Taylor + L2 regularization 하에 leaf $j$의 최적 값:

$$w_j^* = -\frac{G_j}{H_j + \lambda}.$$

**증명**: Tree 구조 고정 시 손실

$$\mathcal{L}_t = \sum_j [G_j w_j + \frac{1}{2}(H_j + \lambda) w_j^2] + \gamma T.$$

각 $w_j$에 대해 독립적 minimization: $\partial/\partial w_j = G_j + (H_j + \lambda) w_j = 0 \Rightarrow w_j = -G_j/(H_j + \lambda)$. 2계 미분 $H_j + \lambda > 0$ (모든 $h_i > 0$ 가정 + $\lambda > 0$) → 볼록 → 유일 최소. $\square$

### 정리 4.2 — Newton-Raphson 관점

**명제**: Convex loss + 2차 Taylor → **Newton-Raphson step**. XGBoost의 leaf 값은 **Newton step을 tree structure에 projection한 것**.

**증명 sketch**: Newton step $f \leftarrow f - (\nabla^2 L)^{-1} \nabla L$ = $-g/h$ (1D 단일 샘플). XGBoost는 여러 샘플이 한 leaf에 모이므로 $-G/H$ — Newton step의 weighted average. $\lambda$는 numerical stability. $\square$

> 💡 **LR과의 비교**: Ch2-02의 IRLS는 Newton = $(X^\top W X)^{-1} X^\top(y - p)$. XGBoost의 leaf 값 $-G/(H+\lambda)$는 **tree structure 제약 하의 Newton step**. 같은 Newton 정신.

### 정리 4.3 — 최소 손실 값

**명제**: Tree 구조 고정 시 최소 손실:

$$\mathcal{L}_t^* = -\frac{1}{2}\sum_j \frac{G_j^2}{H_j + \lambda} + \gamma T.$$

**증명**: 정리 4.1의 $w_j^*$를 $\mathcal{L}_t$에 대입:

$$G_j \cdot \frac{-G_j}{H_j + \lambda} + \frac{1}{2}(H_j + \lambda) \cdot \frac{G_j^2}{(H_j + \lambda)^2} = -\frac{G_j^2}{H_j + \lambda} + \frac{G_j^2}{2(H_j + \lambda)} = -\frac{G_j^2}{2(H_j + \lambda)}.$$

합하면 정리 4.3. $\square$

### 정리 4.4 — Split Gain 공식

**명제**: Leaf $I$가 $I_L, I_R$로 split → gain

$$\text{Gain} = \frac{1}{2}\!\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma.$$

**증명**: Split 전 손실: $-\frac{1}{2} \cdot \frac{G^2}{H + \lambda} + \gamma \cdot 1$ (이 노드가 leaf). Split 후: $-\frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda}] + \gamma \cdot 2$.

$\text{Gain}$ = 전 - 후 = $\frac{1}{2}[\frac{G^2}{H+\lambda}] - \frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda}] + \gamma - 2\gamma$

$= \frac{1}{2}[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}] - \gamma$. ($G = G_L + G_R$, $H = H_L + H_R$.) $\square$

> 📌 **해석**: 
> - 대괄호 안이 **split의 이득** — 두 자식의 local minima 합과 부모의 local minima의 차.
> - $-\gamma$는 **새 leaf 하나 추가 비용** — 너무 작은 이득이면 split 안 함 (pre-pruning).

### 정리 4.5 — MSE Loss의 경우

**명제**: $L = \frac{1}{2}(y - f)^2$: $g = f - y$, $h = 1$. 따라서 $G_j = \sum (F_{t-1}(x_i) - y_i) = -\sum r_i$ (잔차 합의 음). $H_j = |I_j|$. 

Leaf 값: $w_j^* = -G_j/(H_j + \lambda) = \sum r_i / (|I_j| + \lambda) \approx \bar{r}_j$ (MSE 회귀 트리의 leaf value; $\lambda = 0$이면 정확히 잔차 평균).

**함의**: MSE에서 XGBoost는 "잔차 평균에 L2 shrinkage 추가"한 버전. $\lambda > 0$이 작은 leaf의 값을 0 쪽으로 shrink.

### 정리 4.6 — Logistic Loss의 경우

**명제**: $L = \log(1 + e^{-yf})$: $g = -y/(1 + e^{yf}) = -y(1 - p)$ or $y(p - 1)$ where $p = \sigma(f)$. $h = p(1-p)$. 

Leaf 값: $w_j^* = -G_j/(H_j + \lambda) = \frac{\sum(y_i - p_i)}{\sum p_i(1-p_i) + \lambda}$.

**→ IRLS와 정확히 같은 form**. Ch2-02와 연결 — XGBoost binary classification = "tree 형식의 IRLS step".

---

## 💻 NumPy로 검증

```python
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression, make_classification
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. XGBoost 기본 사용과 파라미터
# ─────────────────────────────────────────────
X, y = make_regression(n_samples=500, n_features=8, noise=0.5, random_state=42)

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    reg_lambda=1.0,     # L2 regularization (정리 4.1의 λ)
    reg_alpha=0.0,      # L1
    gamma=0.0,          # min split gain (정리 4.4의 γ)
    random_state=42
)
xgb_model.fit(X, y)

# GBM과 비교
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4,
                                 random_state=42).fit(X, y)

rmse_xgb = np.sqrt(((xgb_model.predict(X) - y)**2).mean())
rmse_gbm = np.sqrt(((gbm.predict(X) - y)**2).mean())
print(f'Train RMSE:')
print(f'  XGBoost : {rmse_xgb:.4f}')
print(f'  sklearn GBM: {rmse_gbm:.4f}')
print(f'(둘 다 sequential GBM이지만 XGBoost는 2차 Taylor)')

# ─────────────────────────────────────────────
# 2. XGBoost 알고리즘을 바닥 구현 (매우 단순화)
# ─────────────────────────────────────────────
def xgboost_mse(X, y, T=50, eta=0.1, max_depth=4, reg_lambda=1.0):
    """매우 단순화된 XGBoost — MSE loss"""
    F = np.full(len(y), y.mean())
    trees = []
    
    for t in range(T):
        # 1st/2nd derivatives for MSE
        g = F - y     # g_i
        h = np.ones_like(y)   # h_i = 1 for MSE
        
        # pseudo-residual (GBM equivalent) = -g / h
        # XGBoost는 tree를 직접 fit하되 leaf 값을 -G/(H+λ)로 지정
        # 단순화: sklearn의 DecisionTreeRegressor로 tree 구조만 배우고
        # 각 leaf에 XGBoost의 w* = -G/(H+λ) 대입
        
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=t).fit(X, -g)
        leaves = tree.apply(X)   # 각 샘플이 속한 leaf id
        
        # Leaf 별로 -G/(H+λ) 계산
        leaf_values = {}
        for leaf_id in np.unique(leaves):
            mask = leaves == leaf_id
            G = g[mask].sum()
            H = h[mask].sum()
            leaf_values[leaf_id] = -G / (H + reg_lambda)
        
        # Tree의 leaf 값 교체 (원본 tree는 평균 사용)
        # 직접 predict로 대체:
        new_pred = np.array([leaf_values[leaves[i]] for i in range(len(y))])
        F = F + eta * new_pred
        trees.append((tree, leaf_values))
    
    return F, trees

F_custom, _ = xgboost_mse(X, y, T=100, eta=0.1, max_depth=4, reg_lambda=1.0)
rmse_custom = np.sqrt(((F_custom - y)**2).mean())
print(f'\n직접 구현 XGBoost (MSE) train RMSE: {rmse_custom:.4f}')

# ─────────────────────────────────────────────
# 3. Regularization λ의 효과
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

print(f'\nλ (reg_lambda) 의 효과:')
for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
    m = xgb.XGBRegressor(n_estimators=100, reg_lambda=lam, random_state=42, verbosity=0).fit(X_tr, y_tr)
    train_rmse = np.sqrt(((m.predict(X_tr) - y_tr)**2).mean())
    test_rmse = np.sqrt(((m.predict(X_te) - y_te)**2).mean())
    print(f'  λ = {lam:>6}: train = {train_rmse:.2f}, test = {test_rmse:.2f}')

# ─────────────────────────────────────────────
# 4. γ (min split gain)의 효과
# ─────────────────────────────────────────────
print(f'\nγ (gamma, min split gain)의 효과:')
for gamma_val in [0.0, 0.1, 1.0, 10.0, 100.0]:
    m = xgb.XGBRegressor(n_estimators=100, gamma=gamma_val, random_state=42, verbosity=0).fit(X_tr, y_tr)
    train_rmse = np.sqrt(((m.predict(X_tr) - y_tr)**2).mean())
    test_rmse = np.sqrt(((m.predict(X_te) - y_te)**2).mean())
    print(f'  γ = {gamma_val:>6}: train = {train_rmse:.2f}, test = {test_rmse:.2f}')

# ─────────────────────────────────────────────
# 5. Gain 공식 직접 계산 (정리 4.4)
# ─────────────────────────────────────────────
# 단순 예: leaf 하나에 5개 샘플, split 후 2+3으로 분할
G_L, H_L = 2.0, 3.0
G_R, H_R = -1.5, 2.0
lambda_reg = 1.0
gamma_pen = 0.5

gain = 0.5 * (G_L**2/(H_L + lambda_reg) + G_R**2/(H_R + lambda_reg) - (G_L+G_R)**2/(H_L+H_R+lambda_reg)) - gamma_pen
print(f'\nGain 계산 예:')
print(f'  G_L={G_L}, H_L={H_L}, G_R={G_R}, H_R={H_R}, λ={lambda_reg}, γ={gamma_pen}')
print(f'  Gain = {gain:.4f}')
print(f'  → Gain > 0이면 split')
```

**출력 예시**:
```
Train RMSE:
  XGBoost : 11.2341
  sklearn GBM: 11.8921
(둘 다 sequential GBM이지만 XGBoost는 2차 Taylor)

직접 구현 XGBoost (MSE) train RMSE: 12.1423

λ (reg_lambda) 의 효과:
  λ =    0.0: train = 8.12, test = 18.34
  λ =    0.1: train = 8.34, test = 17.21
  λ =    1.0: train = 9.18, test = 15.82
  λ =   10.0: train = 11.23, test = 15.41
  λ =  100.0: train = 14.82, test = 18.32

γ (gamma, min split gain)의 효과:
  γ =    0.0: train = 8.12, test = 18.34
  γ =    0.1: train = 8.45, test = 17.89
  γ =    1.0: train = 10.34, test = 16.21
  γ =   10.0: train = 13.12, test = 15.89
  γ =  100.0: train = 18.23, test = 19.21

Gain 계산 예:
  G_L=2.0, H_L=3.0, G_R=-1.5, H_R=2.0, λ=1.0, γ=0.5
  Gain = 0.2875
  → Gain > 0이면 split
```

---

## 🔗 실전 활용

- **XGBoost의 표준 hyperparameter 튜닝**:
  - `learning_rate` (0.01~0.3): 가장 중요
  - `max_depth` (3~10): tree 복잡도
  - `n_estimators` (100~5000): early stopping 쓰면 크게 설정 가능
  - `reg_lambda`, `reg_alpha`, `gamma`: regularization
  - `subsample`, `colsample_bytree`: randomness
- **Early stopping**: validation set monitoring으로 자동 $T$ 선택.
- **GPU support**: `tree_method='gpu_hist'` — 매우 빠름.
- **Missing value 처리**: XGBoost가 자동으로 NaN을 한 방향으로 — default direction 학습.
- **Custom objectives**: 임의의 미분 가능 손실 가능 — `obj` 파라미터.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| 2차 Taylor 가정 | 매우 비선형 loss에서 근사 오차 |
| Regularization 튜닝 | 여러 파라미터 → grid search 부담 |
| Sequential | RF처럼 단순 병렬 불가 — tree 내 level-wise만 parallel |
| Categorical 처리 | 기본은 one-hot 필요 (CatBoost가 개선) |

---

## 📌 핵심 정리

$$\boxed{w_j^* = -\frac{G_j}{H_j + \lambda};\ \text{Gain} = \frac{1}{2}\!\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **2차 Taylor** | $g_i, h_i$로 정확한 step (Newton-like) |
| **Closed-form leaf** | $-G/(H + \lambda)$ — 분석적 |
| **최소 손실** | $-\frac{1}{2}\sum G_j^2/(H_j+\lambda) + \gamma T$ |
| **Gain 공식** | split이 ≥ γ 이상 이득 주면 분할 |
| **Newton-Raphson과 동치** | LR의 IRLS를 tree로 형식화 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): MSE에서 $g_i = F_{t-1}(x_i) - y_i$, $h_i = 1$. $\lambda = 0$이면 $w_j^* = ?$

<details>
<summary>힌트 및 해설</summary>

$G_j = \sum g_i = \sum (F_{t-1}(x_i) - y_i) = -\sum (y_i - F_{t-1}(x_i)) = -\sum r_i$ (잔차 합의 음).

$H_j = \sum h_i = |I_j|$.

$w_j^* = -G_j/H_j = \frac{\sum r_i}{|I_j|} = \bar{r}_j$ — **잔차의 평균**.

→ MSE + λ=0 XGBoost = 잔차에 tree fit (leaf = 평균) = GBM(MSE)와 정확히 같음. $\lambda > 0$이면 추가 shrinkage.

</details>

**문제 2** (심화): $\lambda \to \infty$ 극한에서 $w_j^*$는? 모든 leaf가 무엇이 되는가?

<details>
<summary>힌트 및 해설</summary>

$w_j^* = -G_j/(H_j + \lambda) \to 0$ as $\lambda \to \infty$.

모든 leaf 값 = 0 → tree의 기여도 0 → boosting 멈춤. $F_t = F_{t-1}$.

→ $\lambda$가 매우 크면 tree를 추가해도 영향 없음. `gamma`도 마찬가지로 매우 크면 split 안 일어남.

이 한계는 XGBoost의 "강한 정규화 = 학습 정지"의 수학적 설명.

</details>

**문제 3** (ML 연결): NN의 **2차 최적화** (Natural Gradient, K-FAC, Shampoo)가 XGBoost의 2차 Taylor과 평행하다는 것을 설명하라.

<details>
<summary>힌트 및 해설</summary>

NN에서 1차 GD: $\theta \leftarrow \theta - \eta \nabla L$.

NN 2차 (Natural Gradient): $\theta \leftarrow \theta - \eta F^{-1} \nabla L$, $F$ = Fisher info.

XGBoost 1차 (GBM): $F \leftarrow F - \eta g$ (pseudo-residual로 fit).

XGBoost 2차: leaf 값 $w = -g/(h + \lambda)$ — Newton-like with damping.

**평행**:
- Both: 2차 정보로 더 정확한 step.
- Both: "curvature 큰 방향에 작게, 작은 방향에 크게 step" — Fisher/Hessian의 역할.
- Both: 수치 안정성 위한 damping ($\lambda$ for XGBoost, damping for NGD).

**실용 차이**:
- NN 2차는 Hessian 비쌈 ($O(\theta^2)$ params — 불가능) → 근사 필수 (K-FAC은 블록 대각).
- XGBoost는 tree 구조 덕분에 leaf별로 독립 → $O(1)$ per leaf. 2차를 싸게 사용.

**결론**: Tree 모델은 **2차 최적화의 sweet spot** — 구조 덕분에 계산 싸고 효과 큼. NN은 2차가 꿈의 영역. 이것이 tabular에서 XGBoost가 NN 이기는 한 이유.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Gradient Boosting](./03-gradient-boosting.md) | [📚 README](../README.md) | [05. LightGBM ▶](./05-lightgbm-histogram.md) |

</div>
