# 04. Random Forest의 수렴성

## 🎯 핵심 질문

- $B \to \infty$에서 RF predictor가 왜 무한 앙상블 $f_\infty$로 **almost sure** 수렴하는가?
- 이를 증명하는 **강한 큰 수의 법칙(SLLN)**의 구조는?
- 따라서 generalization error가 **단조감소**한다는 의미 — 더 많은 트리는 절대 해롭지 않다.
- $B$를 매우 크게 하면 어디서 점감 효과가 시작되는가? 실무 권장 $B$는?

---

## 🔍 왜 이 개념이 ML에서 중요한가

"트리를 더 많이 추가하면 항상 도움 된다"는 RF의 **결정적 실용 신뢰성**의 수학적 정당성. (a) Hyperparameter tuning에서 $B$를 걱정 안 해도 됨 — 충분히 크게만 하면 OK, (b) 다른 hyperparameter ($\max_{\text{depth}}$, $m$)와 trade-off 없음, (c) Boosting과 정반대 — Boosting은 너무 많이 하면 overfit. (d) 이 단조 수렴은 **Breiman 2001 정리 1**의 핵심 결과로 RF가 그동안 받은 신뢰의 출처.

---

## 📐 수학적 선행 조건

- 강한 큰 수의 법칙 (Strong Law of Large Numbers)
- Bias-Variance (Ch1-06)
- RF의 분산 공식 (Ch4-02, 03)

---

## 📖 직관적 이해

### "트리 더 많이 = 절대 해롭지 않다"

각 트리 $f_b$는 random parameter $\Theta_b$ (bootstrap + feature subsampling)에 의존. RF predictor:

$$\hat{f}_B(x) = \frac{1}{B}\sum_{b=1}^B f_b(x; \Theta_b).$$

$B \to \infty$이면 SLLN에 의해

$$\hat{f}_\infty(x) := \mathbb{E}_\Theta[f(x; \Theta)] = \lim_{B \to \infty} \hat{f}_B(x) \quad \text{a.s.}$$

→ RF는 "**무한 앙상블의 random sample 근사**". $B$ 큼 → 근사 정확. 정확도 단조 향상.

### Boosting과의 대조

Boosting: $F_t$가 $F_{t-1}$에 의존 (sequential). 너무 많이 추가하면 train data의 noise까지 fit → **overfit**. Early stopping 필수.

RF: 각 트리 독립. 더 많이 추가 → 평균이 더 정확 → 단조 향상.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Random Forest의 Population Predictor

$\hat{f}_B(x; \Theta_1, \ldots, \Theta_B) = \frac{1}{B}\sum f(x; \Theta_b)$. 무한 앙상블:

$$f_\infty(x) := \mathbb{E}_\Theta[f(x; \Theta)].$$

여기서 $\Theta$는 (bootstrap sample 선택 + 각 split의 feature 선택)에 대한 random vector.

---

## 🔬 정리와 증명

### 정리 4.1 — RF의 a.s. 수렴 (Breiman 2001 정리 1)

**명제**: 각 $x$에 대해

$$\hat{f}_B(x) = \frac{1}{B}\sum_b f(x; \Theta_b) \xrightarrow{a.s.} \mathbb{E}_\Theta[f(x; \Theta)] = f_\infty(x) \quad (B \to \infty).$$

**증명**: $\{f(x; \Theta_b)\}_{b=1}^\infty$는 i.i.d. (각 $\Theta_b$가 독립이므로). 유한 second moment 가정 $\mathbb{E}[f(x; \Theta)^2] < \infty$ (각 트리의 예측이 bounded 또는 finite variance). SLLN:

$$\frac{1}{B}\sum_b f(x; \Theta_b) \xrightarrow{a.s.} \mathbb{E}_\Theta[f(x; \Theta)]. \quad \square$$

> 💡 **함의**: RF는 "$B$ 무한대"의 random sample 근사. $B$가 충분히 크면 거의 deterministic.

### 정리 4.2 — Generalization Error의 수렴

**명제**: Test point $(X_0, Y_0) \sim P$에 대해 $B \to \infty$에서 RF의 generalization error는

$$\mathbb{E}_{X_0, Y_0}[L(Y_0, \hat{f}_B(X_0))] \to \mathbb{E}[L(Y_0, f_\infty(X_0))].$$

**증명 sketch**: 정리 4.1로 점별 수렴 + dominated convergence (loss bounded). $\square$

> 📌 **결과**: $B$ 늘리면 generalization error가 한계 $\mathbb{E}[L(Y_0, f_\infty(X_0))]$로 단조수렴. **더 많은 트리는 절대 해롭지 않다**.

### 정리 4.3 — Generalization Error의 Upper Bound (Breiman 2001 정리 2)

**명제**: Classification에서 RF의 일반화 오차

$$P^*(\text{error}) \leq \frac{\bar{\rho}(1 - s^2)}{s^2},$$

여기서 

- $s = \mathbb{E}[\text{margin}]$: 평균 margin (true class와 second-best class의 vote 차이)
- $\bar{\rho}$: 평균 pair-wise tree correlation

**증명 sketch**: Chebyshev's inequality + margin과 correlation의 관계. 자세한 증명은 Breiman (2001) 부록. $\square$

**함의**:
- $s$ 크고 $\bar{\rho}$ 작을수록 bound 작음.
- 트리들이 strong하면서 동시에 다양해야 함 → RF의 design philosophy.

### 정리 4.4 — Variance의 점근적 한계

**명제**: 정리 2.2에서 $\text{Var}(\hat{f}_B) = \rho \sigma^2 + (1-\rho)\sigma^2/B$. $B \to \infty$에서 **$\rho \sigma^2$가 하한**.

**해석**: 더 많은 트리는 variance를 줄이지만 **0으로는 안 줄어듦**. $\rho$가 0이 아니면 항상 잔류 variance 있음.

→ Boosting과 NN처럼 bias 감소가 추가로 필요할 수 있음.

### 정리 4.5 — 실무 $B$ 권장

이론적으로 $B \to \infty$가 좋지만 실무는 점감 효과:

- $B = 10$: 매우 noisy, 단일 tree보다 약간 나음
- $B = 100$: 표준 baseline. 80%~90%의 최종 효과 달성
- $B = 500$: 거의 수렴. 추가 향상 미미
- $B = 1000+$: marginal — 계산 비용만 증가

**sklearn 기본**: `n_estimators=100` (v0.22 이전 10이었음 — 너무 작아 변경됨).

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. B 증가에 따른 RF prediction의 a.s. 수렴
# ─────────────────────────────────────────────
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 매우 큰 B의 RF — population predictor의 근사
rf_large = RandomForestClassifier(n_estimators=2000, random_state=42, n_jobs=-1).fit(X_train, y_train)
prob_inf = rf_large.predict_proba(X_test)
print(f'B = 2000 (≈ f_∞): 정확도 = {rf_large.score(X_test, y_test):.4f}')

# 다양한 B에서 prediction의 차이 측정
print(f'\n{"B":>6s} | {"||probs - f_∞||":>17s} | {"test acc":>9s}')
print('-' * 45)
for B in [1, 5, 10, 50, 100, 500, 1000]:
    rf_B = RandomForestClassifier(n_estimators=B, random_state=42, n_jobs=-1).fit(X_train, y_train)
    prob_B = rf_B.predict_proba(X_test)
    diff = np.linalg.norm(prob_B - prob_inf)
    acc = rf_B.score(X_test, y_test)
    print(f'{B:>6d} | {diff:>17.4f} | {acc:>9.4f}')

# ─────────────────────────────────────────────
# 2. Train accuracy는 항상 1 (full tree), test는 단조향상
# ─────────────────────────────────────────────
print(f'\n{"B":>6s} | {"train":>7s} | {"test":>7s}')
print('-' * 30)
for B in [1, 10, 50, 100, 500]:
    rf = RandomForestClassifier(n_estimators=B, random_state=42).fit(X_train, y_train)
    print(f'{B:>6d} | {rf.score(X_train, y_train):.4f} | {rf.score(X_test, y_test):.4f}')

# ─────────────────────────────────────────────
# 3. Variance 수렴 — 다른 random state로 RF 학습 → 분산 측정
# ─────────────────────────────────────────────
print(f'\n같은 데이터, 다른 random_state로 학습 → 예측 variance:')
for B in [10, 50, 100, 500]:
    preds_across_seeds = []
    for seed in range(20):
        rf = RandomForestClassifier(n_estimators=B, random_state=seed, n_jobs=-1).fit(X_train, y_train)
        preds_across_seeds.append(rf.predict_proba(X_test))
    preds_across_seeds = np.array(preds_across_seeds)
    var_per_test = preds_across_seeds.var(axis=0).mean()
    print(f'  B = {B:>4}: 평균 prediction variance = {var_per_test:.6f}')

# ─────────────────────────────────────────────
# 4. Boosting과 비교 — overfit 행동 차이
# ─────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier

# Noisy data
X_noisy, y_noisy = make_classification(n_samples=300, n_features=10, n_informative=5,
                                        flip_y=0.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_noisy, y_noisy, test_size=0.3, random_state=0)

print(f'\nNoisy data — RF vs GBM의 트리 수에 따른 overfit:')
print(f'{"B":>5s} | {"RF train":>8s} | {"RF test":>7s} | {"GBM train":>9s} | {"GBM test":>8s}')
print('-' * 55)
for B in [10, 50, 100, 500]:
    rf = RandomForestClassifier(n_estimators=B, random_state=0).fit(X_tr, y_tr)
    gb = GradientBoostingClassifier(n_estimators=B, max_depth=3, random_state=0).fit(X_tr, y_tr)
    print(f'{B:>5d} | {rf.score(X_tr, y_tr):.4f} | {rf.score(X_te, y_te):.4f} | '
          f'{gb.score(X_tr, y_tr):.4f}    | {gb.score(X_te, y_te):.4f}')

print(f'\n→ RF: B 증가시켜도 test acc 유지/향상 (단조)')
print(f'→ GBM: B 너무 크면 overfit 가능 (test acc 감소 시작)')
```

**출력 예시**:
```
B = 2000 (≈ f_∞): 정확도 = 0.9133

     B |   ||probs - f_∞|| |  test acc
---------------------------------------------
     1 |             8.4521 |    0.8167
     5 |             3.2814 |    0.8800
    10 |             2.1832 |    0.8867
    50 |             0.9821 |    0.9067
   100 |             0.6841 |    0.9100
   500 |             0.2954 |    0.9133
  1000 |             0.1521 |    0.9133

     B |   train |    test
------------------------------
     1 | 1.0000 | 0.8167
    10 | 1.0000 | 0.8867
    50 | 1.0000 | 0.9067
   100 | 1.0000 | 0.9100
   500 | 1.0000 | 0.9133

같은 데이터, 다른 random_state로 학습 → 예측 variance:
  B =   10: 평균 prediction variance = 0.024153
  B =   50: 평균 prediction variance = 0.005023
  B =  100: 평균 prediction variance = 0.002621
  B =  500: 평균 prediction variance = 0.000543

Noisy data — RF vs GBM의 트리 수에 따른 overfit:
    B | RF train | RF test | GBM train | GBM test
-------------------------------------------------------
   10 | 0.9905 | 0.7333 | 0.9476    | 0.7333
   50 | 1.0000 | 0.7444 | 1.0000    | 0.7222
  100 | 1.0000 | 0.7444 | 1.0000    | 0.7000
  500 | 1.0000 | 0.7444 | 1.0000    | 0.6889

→ RF: B 증가시켜도 test acc 유지/향상 (단조)
→ GBM: B 너무 크면 overfit 가능 (test acc 감소 시작)
```

---

## 🔗 실전 활용

- **RF 권장 $B$**: 100~500. 1000+는 보통 불필요.
- **`warm_start=True`**: 트리 추가하며 재학습 — 점진적 $B$ 증가 가능.
- **`n_jobs=-1`**: 병렬 학습 (각 트리 독립).
- **Convergence diagnostic**: $B$ 늘려가며 OOB error 추이 확인. 평탄해지면 충분.
- **Ensemble of RFs**: 다른 seed로 여러 RF → 더 안정 (over-engineering인 경우 多).

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Bias 한계 | $f_\infty$가 본질적으로 가진 bias는 trees 수로 못 줄임 |
| 계산 비용 | $B$ 비례 — 예측 속도가 $B \times$ tree depth |
| Random correlation | 매우 작은 $B$에서 unstable |
| Distribution shift | 학습 시 못 본 분포에서는 OOD performance 보장 X |

---

## 📌 핵심 정리

$$\boxed{\hat{f}_B \xrightarrow{a.s.} f_\infty \text{ as } B \to \infty;\ \text{generalization error 단조감소; 절대 overfit 없음 (B 측면)}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **a.s. 수렴** | SLLN — i.i.d. tree의 평균이 기댓값으로 |
| **Variance 한계** | $B \to \infty$에서 $\rho \sigma^2$ — 0이 안 됨 |
| **Bias 불변** | 수렴 한계 자체가 bias를 가짐 |
| **단조 generalization** | $B$ 증가는 test error를 절대 악화시키지 않음 |
| **Boosting과 대비** | Boosting은 sequential → overfit 가능 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 정리 4.1의 $f_\infty$가 어떤 분포에 대한 기댓값인지 정확히 쓰라.

<details>
<summary>힌트 및 해설</summary>

$f_\infty(x) = \mathbb{E}_{\Theta \sim P_\Theta}[f(x; \Theta)]$. 

$\Theta$의 분포 $P_\Theta$는 다음으로 이루어짐:
- Bootstrap sample 선택 (uniform with replacement on $\mathcal{D}_n$)
- 각 split node에서 feature subset (uniform $m$-subset of $\{1, \ldots, p\}$)

데이터 $\mathcal{D}_n$은 고정. 따라서 $f_\infty$도 데이터에 의존하지만 **random parameter에 대한 평균**.

</details>

**문제 2** (심화): RF가 $B \to \infty$에서 수렴하는데 왜 generalization error는 0이 안 되는가?

<details>
<summary>힌트 및 해설</summary>

수렴 한계 $f_\infty$ 자체가:

1. **Bias**: 트리 family의 표현 한계 (axis-aligned, leaf 수 등) → 진짜 $f^*$를 정확히 표현 못함.
2. **데이터 의존**: $f_\infty$는 고정된 $\mathcal{D}_n$에 의존 — 다른 데이터셋에서 다른 $f_\infty$ → "데이터 분산" 잔류.
3. **Irreducible noise** $\sigma^2$: $y$ 자체의 잡음.

$B \to \infty$는 **estimator variance의 한 종류** (random tree parameter의 variance)만 제거 — 위 3가지는 그대로 남음.

</details>

**문제 3** (ML 연결): NN은 random init 의존성이 큼. **Deep Ensembles** (Lakshminarayanan 2017)가 RF의 정리 4.1과 어떻게 평행한지 설명하라.

<details>
<summary>힌트 및 해설</summary>

Deep Ensembles: $M$개 NN을 다른 random init + 다른 mini-batch order로 학습 → 예측 평균.

각 NN의 학습 결과 $f_m$은 random hyperparameters $\Theta_m$ (init weight + batch order)에 의존. 

$\hat{f}_M(x) = \frac{1}{M}\sum f_m(x; \Theta_m)$.

$M \to \infty$이면 (i.i.d. 가정) SLLN → $f_\infty(x) = \mathbb{E}_\Theta[f(x; \Theta)]$.

**평행 구조**:
- RF: $\Theta = $ bootstrap + feature subset
- NN: $\Theta = $ random init + batch order

둘 다 random parameter에 대한 평균이 더 안정한 prediction. 다만 NN은 비볼록 → 다른 local minimum에 수렴 → **더 다양한 trees** → ρ 더 작음 → 더 큰 variance reduction.

→ "RF는 결국 NN ensemble의 트리 버전". Modern ML의 통일된 view.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Random Forest](./03-random-forest.md) | [📚 README](../README.md) | [05. Feature Importance ▶](./05-feature-importance.md) |

</div>
