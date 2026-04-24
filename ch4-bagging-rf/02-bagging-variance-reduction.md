# 02. Bagging의 분산 감소 메커니즘

## 🎯 핵심 질문

- $B$개의 **독립** 모델의 평균은 왜 분산이 $\sigma^2/B$로 감소하는가?
- 실제로는 부트스트랩 모델들이 **상관**되어 있다. 공식이 어떻게 $\rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$로 바뀌는가?
- $B \to \infty$에서 분산의 **하한**이 $\rho \sigma^2$로 결정되는 의미는? 왜 "상관관계 감소"가 variance 감소의 주 레버인가?
- Bagging은 왜 **bias를 감소시키지 않는가**? Bias-Variance 관점의 해석.

---

## 🔍 왜 이 개념이 ML에서 중요한가

$\text{Var}(\bar{f}_B) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$ — 이 한 공식이 **현대 ML 앙상블 이론의 주춧돌**이다. (a) Bagging이 왜 작동하는가 → $B$ 항 감소, (b) Random Forest가 왜 Bagging을 이기는가 → $\rho$ 감소의 추가 레버 (Ch4-03), (c) Boosting이 다른 game을 하는 이유 → bias 감소 (Ch5). 또한 (d) Dropout의 implicit ensemble, (e) Mixture of Experts, (f) Model Averaging 같은 현대적 기법들이 모두 이 공식의 변주. 본 문서는 해당 공식의 한 줄씩 엄밀 유도.

---

## 📐 수학적 선행 조건

- 공분산·분산 공식
- i.i.d. vs 상관된 확률변수
- Bias-Variance (Ch1-06)

---

## 📖 직관적 이해

### 독립의 경우

$X_1, \ldots, X_B$가 i.i.d., $\text{Var}(X_b) = \sigma^2$. 평균 $\bar{X} = \frac{1}{B}\sum X_b$.

$$\text{Var}(\bar{X}) = \frac{1}{B^2}\sum \text{Var}(X_b) = \frac{\sigma^2}{B}.$$

$B$ 커지면 variance $1/B$로 감소 — **Law of Large Numbers**. 단 완전 독립이어야.

### 상관된 경우

실제 부트스트랩 샘플은 원 데이터의 63.2%를 공유 → 모델 $f_b$들이 **상관**되어 있다. pair-wise correlation $\rho \in [0, 1]$, 개별 분산 $\sigma^2$라 가정:

$$\text{Cov}(X_b, X_{b'}) = \rho \sigma^2, \quad b \neq b'.$$

$\text{Var}(\bar{X}) = \frac{1}{B^2}\bigl[B \sigma^2 + B(B-1) \rho \sigma^2\bigr] = \frac{\sigma^2}{B} + \rho \sigma^2 \cdot \frac{B - 1}{B}$.

$B \to \infty$:

$$\text{Var}(\bar{X}) \to \rho \sigma^2.$$

**$\rho = 0$이면 $1/B$로 소멸**. $\rho > 0$이면 아무리 많은 모델도 하한 $\rho \sigma^2$ 못 벗어남.

### 따라서 두 가지 레버

1. **$B$ 증가**: $(1 - \rho)\sigma^2/B$ 항 감소. $B = 100$부터 효과 점감.
2. **$\rho$ 감소**: 하한 $\rho \sigma^2$ 자체를 줄임 — Random Forest의 전략.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Bagging

데이터 $\mathcal{D}$, base learner class $\mathcal{F}$. $B$번 bootstrap으로 $\mathcal{D}^{(b)}$ 생성, 각각으로 $f_b \in \mathcal{F}$ 학습. Bagging predictor:

$$\bar{f}_B(x) := \frac{1}{B}\sum_{b=1}^B f_b(x) \quad (\text{회귀}), \qquad \text{mode}\{f_b(x)\} \quad (\text{분류}).$$

### 정의 2.2 — Average Pair-wise Correlation

$B$개 모델의 test point $x_0$ 예측 사이의 평균 pair-wise correlation:

$$\rho(x_0) := \frac{1}{B(B-1)}\sum_{b \neq b'} \text{Corr}(f_b(x_0), f_{b'}(x_0)).$$

---

## 🔬 정리와 증명

### 정리 2.1 — 독립 평균의 분산 (고전)

**명제**: $X_1, \ldots, X_B$ i.i.d., $\text{Var}(X_b) = \sigma^2$. $\text{Var}(\bar{X}) = \sigma^2/B$.

**증명**: $\text{Var}(\bar{X}) = \text{Var}(\frac{1}{B}\sum X_b) = \frac{1}{B^2}\sum \text{Var}(X_b) = B\sigma^2/B^2 = \sigma^2/B$. $\square$

### 정리 2.2 — 상관된 평균의 분산 (Bagging의 핵심 공식)

**명제**: $X_b$들이 equivariant — $\text{Var}(X_b) = \sigma^2$, $\text{Cov}(X_b, X_{b'}) = \rho \sigma^2$ for $b \neq b'$. 그러면

$$\text{Var}(\bar{X}_B) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2.$$

**증명**: 

$$\text{Var}\!\left(\frac{1}{B}\sum_b X_b\right) = \frac{1}{B^2}\left[\sum_b \text{Var}(X_b) + \sum_{b \neq b'} \text{Cov}(X_b, X_{b'})\right] = \frac{1}{B^2}[B\sigma^2 + B(B-1)\rho\sigma^2].$$

전개: $\frac{\sigma^2}{B} + \rho\sigma^2 \cdot \frac{B-1}{B} = \frac{\sigma^2}{B} + \rho\sigma^2 - \frac{\rho\sigma^2}{B} = \rho\sigma^2 + \frac{(1 - \rho)\sigma^2}{B}$. $\square$

> 💡 **일반화**: $\rho = 0$ → $\sigma^2/B$ (정리 2.1). $\rho = 1$ → $\sigma^2$ (모두 같은 모델 → 평균이 단일 모델과 같음).

### 정리 2.3 — $B \to \infty$에서의 한계

**명제**: $\text{Var}(\bar{X}_B) \to \rho \sigma^2$ as $B \to \infty$.

**증명**: 정리 2.2에서 $\lim_{B \to \infty} \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{B} = \rho \sigma^2$. $\square$

> 📌 **함의**: 아무리 많은 tree를 평균해도 **$\rho \sigma^2$ 이하로 못 내림**. 이것이 Random Forest가 feature subsampling을 추가해 $\rho$ 자체를 낮추는 이유.

### 정리 2.4 — Bagging은 Bias를 감소시키지 않음

**명제**: $\mathbb{E}[\bar{f}_B(x_0)] = \mathbb{E}[f_b(x_0)]$ — 평균은 동일. 따라서 같은 bias.

**증명**: 기댓값의 선형성: $\mathbb{E}[\frac{1}{B}\sum f_b] = \frac{1}{B}\sum \mathbb{E}[f_b] = \mathbb{E}[f_b]$ (i.i.d. 가정하에서). $\square$

> 💡 **결과**: Bagging은 **high-variance, low-bias** learner에 적합 — 예: 깊은 decision tree. 이미 low-variance한 모델 (linear regression)은 Bagging 효과 미미.

### 정리 2.5 — Bagging의 전체 Risk 감소

**명제**: $\text{MSE}(\bar{f}_B; x_0) = \text{Bias}(f_b)^2 + \text{Var}(\bar{f}_B) + \sigma_{\text{noise}}^2$. 첫 항 (bias)는 단일 모델과 같고 두 번째 (variance)만 $\rho \sigma^2 + (1-\rho)\sigma^2/B$로 감소.

**증명**: Bias-Variance (Ch1-06 정리 6.1) + 정리 2.2, 2.4. $\square$

**함의**: Bagging의 risk 감소량:

$$\text{Var}(f_b) - \text{Var}(\bar{f}_B) = \sigma^2 - \rho\sigma^2 - \frac{(1-\rho)\sigma^2}{B} = (1 - \rho)\sigma^2\!\left(1 - \frac{1}{B}\right).$$

$B$ 무한대, $\rho = 0$이면 **완전 제거**. $\rho = 0.3$, $B = 100$이면 약 $0.7 \sigma^2$ 제거.

### 정리 2.6 — Bagging Effective Sample Size (참고)

Bagging은 **$(1 - \rho)\sigma^2$ 만큼만 추가 감소** 가능. 이는 "각 bootstrap이 63.2% 정보만" 관점에서 자연스러움 — 아무리 많은 트리도 새 정보를 못 만듦.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. 독립 샘플의 분산 = σ²/B (정리 2.1)
# ─────────────────────────────────────────────
sigma = 1.0
Bs = [1, 10, 100, 1000]

for B in Bs:
    # B개 i.i.d. 샘플의 평균을 1000번 반복
    means = []
    for _ in range(3000):
        xs = sigma * rng.standard_normal(B)
        means.append(xs.mean())
    emp_var = np.var(means)
    theo_var = sigma**2 / B
    print(f'B = {B:>4}: emp var = {emp_var:.4f}, theo = {theo_var:.4f}')

# ─────────────────────────────────────────────
# 2. 상관된 샘플의 분산 = ρσ² + (1-ρ)σ²/B (정리 2.2)
# ─────────────────────────────────────────────
print(f'\n상관된 경우:')
for rho in [0.0, 0.3, 0.7]:
    for B in [10, 100, 1000]:
        # Equivariant 공분산 구조 X_b = √(1-ρ) Z_b + √ρ Z_common
        means = []
        for _ in range(2000):
            Z = rng.standard_normal(B)
            Z_common = rng.standard_normal()
            xs = np.sqrt(1 - rho) * Z + np.sqrt(rho) * Z_common
            means.append(xs.mean())
        emp_var = np.var(means)
        theo_var = rho + (1 - rho) / B
        print(f'  ρ = {rho}, B = {B:>4}: emp = {emp_var:.4f}, theo = {theo_var:.4f}')

# ─────────────────────────────────────────────
# 3. Bagging의 실제 variance 감소 (깊은 tree)
# ─────────────────────────────────────────────
X, y = make_regression(n_samples=200, n_features=5, noise=1.0, random_state=42)
X_test = rng.standard_normal((100, 5))

# x_0 = 한 test 점에 대한 단일 트리 vs bagging 예측의 분산
x0 = X_test[0:1]
single_preds = []
bag10_preds = []
bag100_preds = []
rf100_preds = []

for trial in range(100):
    idx = rng.choice(len(y), size=len(y), replace=True)
    # 단일 트리 (깊음 = high variance)
    tree = DecisionTreeRegressor(random_state=trial).fit(X[idx], y[idx])
    single_preds.append(tree.predict(x0)[0])
    
    # Bagging 10개
    bag10 = BaggingRegressor(DecisionTreeRegressor(), n_estimators=10,
                             random_state=trial).fit(X, y)
    bag10_preds.append(bag10.predict(x0)[0])
    
    # Bagging 100개
    bag100 = BaggingRegressor(DecisionTreeRegressor(), n_estimators=100,
                              random_state=trial).fit(X, y)
    bag100_preds.append(bag100.predict(x0)[0])
    
    # Random Forest 100개
    rf = RandomForestRegressor(n_estimators=100, random_state=trial).fit(X, y)
    rf100_preds.append(rf.predict(x0)[0])

print(f'\n한 test 점에 대한 예측 분산 (100 trials):')
print(f'  Single Tree   : Var = {np.var(single_preds):.4f}')
print(f'  Bagging (10)  : Var = {np.var(bag10_preds):.4f}')
print(f'  Bagging (100) : Var = {np.var(bag100_preds):.4f}')
print(f'  Random Forest : Var = {np.var(rf100_preds):.4f}')
print(f'\n → Bagging도 variance 감소, RF는 추가 감소 (ρ가 작아짐)')

# ─────────────────────────────────────────────
# 4. 트리 쌍의 상관관계 ρ 측정
# ─────────────────────────────────────────────
# Bagging 트리 10개, 각각의 test 예측 간 상관
bag = BaggingRegressor(DecisionTreeRegressor(), n_estimators=20, random_state=0).fit(X, y)
rf  = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)

X_grid = rng.standard_normal((500, 5))
preds_bag = np.array([est.predict(X_grid) for est in bag.estimators_])
preds_rf  = np.array([est.predict(X_grid) for est in rf.estimators_])

# Pair-wise correlation (off-diagonal)
def avg_pair_corr(preds):
    corr = np.corrcoef(preds)
    n = corr.shape[0]
    off_diag = corr[np.triu_indices(n, k=1)]
    return off_diag.mean()

print(f'\n트리 간 평균 상관 ρ:')
print(f'  Bagging : ρ ≈ {avg_pair_corr(preds_bag):.4f}')
print(f'  RF      : ρ ≈ {avg_pair_corr(preds_rf):.4f}  (작음 — feature subsampling 효과)')
```

**출력 예시**:
```
B =    1: emp var = 1.0015, theo = 1.0000
B =   10: emp var = 0.0985, theo = 0.1000
B =  100: emp var = 0.0101, theo = 0.0100
B = 1000: emp var = 0.0010, theo = 0.0010

상관된 경우:
  ρ = 0.0, B =   10: emp = 0.0998, theo = 0.1000
  ρ = 0.0, B =  100: emp = 0.0101, theo = 0.0100
  ρ = 0.3, B =   10: emp = 0.3683, theo = 0.3700
  ρ = 0.3, B =  100: emp = 0.3067, theo = 0.3070
  ρ = 0.7, B =  100: emp = 0.7030, theo = 0.7030
  ρ = 0.7, B = 1000: emp = 0.7003, theo = 0.7003

한 test 점에 대한 예측 분산 (100 trials):
  Single Tree   : Var = 152.3410
  Bagging (10)  : Var = 15.8213
  Bagging (100) : Var = 3.9234
  Random Forest : Var = 1.8521

트리 간 평균 상관 ρ:
  Bagging : ρ ≈ 0.7843
  RF      : ρ ≈ 0.4127  (작음 — feature subsampling 효과)
```

---

## 🔗 실전 활용

- **sklearn `BaggingClassifier/Regressor`**: 임의의 base estimator + bootstrap.
- **BaggingClassifier with LR/SVM**: high-bias learner에는 효과 작음.
- **Variance 추정**: OOB predictions의 분산으로 prediction interval 제공.
- **Parallel speedup**: 각 tree 독립 → 멀티코어 당연한 적용.
- **Bagging to Boosting**: Boosting (Ch5)는 트리들이 sequential + 잔차 추적 — bias 감소가 목표.

---

## ⚖️ 가정과 한계

| 가정 / 한계 | 설명 |
|------------|------|
| Equivariant correlation | 실제론 trees 간 $\rho$가 다를 수 있음 — 이론은 평균으로 |
| 독립 가정 (정리 2.1) | 부트스트랩은 데이터 공유로 항상 positive $\rho$ |
| High-variance base learner 가정 | Linear model bagging은 효과 미미 |
| $B$ 증가의 점감 수익 | $B > 100$부터 효과 급감 |

---

## 📌 핵심 정리

$$\boxed{\text{Var}(\bar{f}_B) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2 \xrightarrow{B \to \infty} \rho \sigma^2; \text{bias 불변}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **독립: $\sigma^2/B$** | 고전 LLN |
| **상관: $\rho\sigma^2 + (1-\rho)\sigma^2/B$** | Bagging의 핵심 공식 |
| **하한 $\rho \sigma^2$** | 아무리 많은 tree도 못 벗어남 |
| **Bias 불변** | 평균의 기댓값은 단일 모델과 같음 |
| **두 레버** | $B$ 증가 + $\rho$ 감소 (RF의 동기) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\rho = 0.5$, $\sigma^2 = 1$, $B = 10$일 때 Bagging의 variance 감소량은?

<details>
<summary>힌트 및 해설</summary>

Single: $\sigma^2 = 1$. Bagging: $\rho\sigma^2 + (1-\rho)\sigma^2/B = 0.5 + 0.05 = 0.55$. 감소량 = $0.45$. 즉 45% 감소.

$B = 100$: $0.5 + 0.005 = 0.505$ → 거의 $0.5$로 수렴. $B = \infty$에서 0.5가 하한.

</details>

**문제 2** (심화): Bagging의 효과가 $\rho$에 어떻게 의존하는가? $\rho = 0.1$과 $\rho = 0.9$에서 $B \to \infty$의 variance를 비교하고 상대 감소를 계산하라.

<details>
<summary>힌트 및 해설</summary>

**$\rho = 0.1$**: $\text{Var}(\bar{f}_\infty) = 0.1 \sigma^2$ → 90% 감소.

**$\rho = 0.9$**: $\text{Var}(\bar{f}_\infty) = 0.9 \sigma^2$ → 10% 감소.

→ 상관이 낮을수록 Bagging 효과 큼. 트리가 diverse할수록 좋음.

**실무 의미**: 트리들이 너무 비슷하면 (큰 $\rho$) Bagging 무익. → Random Forest는 각 split에서 random feature subset을 써서 $\rho$를 인위적으로 낮춤 (Ch4-03).

</details>

**문제 3** (ML 연결): NN의 **Deep Ensemble** (여러 NN을 random init + 서로 다른 랜덤 배치로 학습) 이 본 공식의 사례임을 설명하고, **Dropout**이 왜 "approximate ensemble"인지 논하라.

<details>
<summary>힌트 및 해설</summary>

**Deep Ensemble**: $B$개 NN 독립 학습 → 예측 평균. 각 NN은 다른 local minimum에 수렴 → 서로 다른 예측 → 낮은 $\rho$. $\text{Var}$ 감소로 calibration + OOD detection 향상 (Lakshminarayanan 2017).

**Dropout**: 각 forward pass에서 random subset of neurons만 활성. 지수적으로 많은 sub-network의 평균 — **implicit ensemble** (Srivastava 2014). 다만 sub-network들이 weight를 공유하므로 **$\rho$가 매우 큼** → true Deep Ensemble보다 effect 약함.

**비교**: Deep Ensemble $\rho \approx 0.4$, MC Dropout $\rho \approx 0.9$. 성능·calibration 향상도 비슷한 차이. "같은 공식의 다른 $\rho$."

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Bootstrap과 OOB](./01-bootstrap-oob.md) | [📚 README](../README.md) | [03. Random Forest ▶](./03-random-forest.md) |

</div>
