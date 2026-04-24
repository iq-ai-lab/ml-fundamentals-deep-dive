# 03. Random Forest의 추가 무작위성

## 🎯 핵심 질문

- 각 split에서 **random feature subset**을 쓰면 왜 트리 간 상관 $\rho$가 감소하는가?
- Classification: $m = \sqrt{p}$, Regression: $m = p/3$의 표준 권장값은 어디서 나왔는가?
- $m$을 더 작게 하면 무엇이 trade-off되는가? 각 트리의 개별 성능 vs 앙상블의 다양성.
- 정리 2.2의 공식에서 **$\rho$ 감소의 제한** — RF도 완벽하지 않은 이유.

---

## 🔍 왜 이 개념이 ML에서 중요한가

Random Forest는 Bagging의 **한 단계 진화** — 트리들의 상관 $\rho$를 낮춰 variance 감소의 상한을 더 내린다. 이 아이디어는 (a) **tabular data의 대표 baseline** — sklearn `RandomForestClassifier`, (b) **Kaggle·실무의 go-to model** — GBM과 함께 최상위권, (c) **trees의 축정렬 편향 + 불안정성** (Ch3-05)을 앙상블로 elegant하게 해결, (d) Breiman의 대표 업적 (2001). 본 문서는 $m$이 왜 정확히 그 값이어야 하는지, 그리고 feature subsampling이 어떻게 "다양성 ≠ 정확성"의 문제를 만드는지 다룬다.

---

## 📐 수학적 선행 조건

- Bagging의 variance 공식 (Ch4-02)
- 결정트리의 분할 (Ch3-01~03)
- Bias-Variance trade-off (Ch1-06)

---

## 📖 직관적 이해

### Bagging의 한계

Bagging 트리들은 **같은 greedy 알고리즘**으로 학습 → 대부분 같은 첫 분할 (가장 IG 큰 feature) → 트리 구조가 비슷 → $\rho$ 큼 ($\approx 0.7$ 정도).

### Random Feature Selection

각 split에서 $p$개 feature 중 **random $m$개만** 후보로 고려. 가장 중요한 feature가 선택 안 될 수도 있음 → 트리들이 **다른 경로**로 분기 → 구조 다양성 ↑ → $\rho$ ↓.

### 왜 $\sqrt{p}$?

**Classification**: $m = \sqrt{p}$가 경험적으로 최적 (Breiman 2001).
- $m = 1$: 너무 random → 개별 tree 성능 나쁨.
- $m = p$: Bagging과 같음 → $\rho$ 큼.
- $\sqrt{p}$는 **다양성 + 개별 정확성의 sweet spot**.

**Regression**: $m = p/3$.

(정확한 이론적 유도보다 대규모 실험적 결과, Breiman 2001 및 후속 연구.)

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Random Forest 알고리즘

1. $B$번 반복:
   a. Bootstrap $\mathcal{D}^{(b)}$.
   b. 트리 $f_b$ 학습:
      - 각 내부 노드에서 $p$개 feature 중 **random $m$개 선택**.
      - 그 $m$개 중 최적 split 선택.
      - 가지치기 안 함 (full grow).
2. 예측:
   - 회귀: $\bar{f}(x) = \frac{1}{B}\sum f_b(x)$.
   - 분류: majority vote 또는 averaged probabilities.

### 정의 3.2 — Tree Correlation

$\rho(x_0) := \text{Corr}(f_b(x_0), f_{b'}(x_0))$ — 두 독립 학습된 tree의 예측 간 상관.

---

## 🔬 정리와 증명

### 정리 3.1 — Feature Subsampling이 $\rho$를 감소시킴

**명제** (informal): RF의 $\rho$ < Bagging의 $\rho$ (같은 $B$, 같은 base learner type). 

**증명 sketch**: Bagging의 두 트리는 같은 시점에서 같은 최적 feature를 고를 확률이 높음 → 구조 유사. RF에서는 각 트리가 다른 random feature subset을 보므로 서로 다른 feature로 분기 → 구조 다양. 형식적 확률 bound는 Breiman (2001) Theorem 2. $\square$

### 정리 3.2 — RF의 Variance (정리 2.2 + 감소된 ρ)

**명제**: RF의 variance

$$\text{Var}(\bar{f}^{RF}) = \rho_{RF} \sigma_{RF}^2 + \frac{1 - \rho_{RF}}{B}\sigma_{RF}^2.$$

Bagging 대비 $\rho_{RF} < \rho_{\text{Bag}}$. 그러나 보통 $\sigma_{RF}^2 > \sigma_{\text{Bag}}^2$ (개별 트리 약간 더 약함).

**총 효과**: $\rho$ 감소가 $\sigma^2$ 증가보다 큼 → RF가 Bagging 보다 더 낮은 variance.

### 정리 3.3 — $m$의 Trade-off

**명제**: $m$ 감소 → (a) $\rho_{RF} \downarrow$ (좋음), (b) $\sigma_{RF}^2 \uparrow$ (나쁨, 각 트리가 best feature를 못 보는 경우 증가).

**최적 $m$**: 두 효과의 balance. $m = \sqrt{p}$ (classification), $p/3$ (regression)이 광범위한 실험에서 관찰된 최적.

**Hypertuning**: sklearn `max_features` — $m$ 조절. $m = 'sqrt'$가 기본값 (classifier).

### 정리 3.4 — RF의 Bias

**명제**: RF의 bias는 Bagging·단일 트리와 비슷 ($\mathbb{E}[\bar{f}^{RF}]$는 개별 $\mathbb{E}[f_b]$의 평균이지만, $f_b$가 살짝 더 "constrained" — 때론 더 큰 bias).

**함의**: RF는 **variance 감소에 특화**. bias 감소는 기대하지 말 것. Boosting (Ch5)이 bias 감소의 역할.

### 정리 3.5 — Generalization Error Bound (Breiman 2001)

**명제** (Breiman 2001): RF의 일반화 오차는

$$P\!\left(\text{error}\right) \leq \frac{\bar{\rho}(1 - s^2)}{s^2},$$

여기서 $s$는 평균 margin (strength), $\bar{\rho}$는 평균 correlation. 

**함의**:
- $s$ 크고 $\bar{\rho}$ 작으면 오차 bound 작음.
- 각 tree가 정확하고(strength 큼) 동시에 서로 다름(correlation 작음) 이상적.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. max_features의 영향 (정리 3.3)
# ─────────────────────────────────────────────
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                            random_state=42)
p = X.shape[1]

print(f'Total features: {p}')
print(f'{"m":>8s} | {"CV acc":>8s} | {"std":>6s}')
print('-' * 30)
for m in [1, 2, 4, int(np.sqrt(p)), 8, 15, p]:
    rf = RandomForestClassifier(n_estimators=100, max_features=m, random_state=42)
    scores = cross_val_score(rf, X, y, cv=5)
    label = f'{m}' if m != int(np.sqrt(p)) else f'{m} (√p)'
    print(f'{label:>8s} | {scores.mean():.4f} | {scores.std():.4f}')

# ─────────────────────────────────────────────
# 2. Bagging vs RF의 tree correlation (정리 3.1)
# ─────────────────────────────────────────────
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=0).fit(X, y)
rf  = RandomForestClassifier(n_estimators=50, max_features='sqrt', random_state=0).fit(X, y)

# 각 tree의 test 예측
X_test, _ = make_classification(n_samples=500, n_features=20, random_state=1)
preds_bag = np.array([est.predict(X_test) for est in bag.estimators_])
preds_rf  = np.array([est.predict(X_test) for est in rf.estimators_])

def avg_pair_corr(preds):
    """정확도 일치 비율 (~correlation)"""
    n = len(preds)
    pairs = 0
    agree = 0
    for i in range(n):
        for j in range(i+1, n):
            agree += (preds[i] == preds[j]).sum()
            pairs += 1
    return agree / (pairs * preds.shape[1])

print(f'\n트리 쌍 간 예측 일치율:')
print(f'  Bagging : {avg_pair_corr(preds_bag):.4f}')
print(f'  RF      : {avg_pair_corr(preds_rf):.4f}  (낮음 — 다양성 증가)')

# ─────────────────────────────────────────────
# 3. RF를 NumPy로 바닥부터 구현
# ─────────────────────────────────────────────
class SimpleRF:
    def __init__(self, n_estimators=100, max_features='sqrt', random_state=0):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
    
    def fit(self, X, y):
        n, p = X.shape
        m = int(np.sqrt(p)) if self.max_features == 'sqrt' else self.max_features
        self.trees_ = []
        rng_local = np.random.default_rng(self.random_state)
        for b in range(self.n_estimators):
            # Bootstrap
            idx = rng_local.integers(0, n, size=n)
            # Random feature subset — 트리 전체에 한 번? No — sklearn은 각 split에 적용
            # 여기서는 트리별 max_features를 sklearn에 전달
            tree = DecisionTreeClassifier(max_features=m, random_state=b)
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self
    
    def predict(self, X):
        probs = np.mean([t.predict_proba(X) for t in self.trees_], axis=0)
        return np.argmax(probs, axis=1)

my_rf = SimpleRF(n_estimators=100, max_features='sqrt', random_state=42).fit(X, y)
sk_rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                random_state=42).fit(X, y)

print(f'\n직접 구현 RF accuracy : {(my_rf.predict(X) == y).mean():.4f}')
print(f'sklearn RF accuracy   : {sk_rf.score(X, y):.4f}')

# ─────────────────────────────────────────────
# 4. 다른 데이터셋 — breast cancer에서 RF vs Bagging
# ─────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X_bc, y_bc = data.data, data.target

from sklearn.model_selection import cross_val_score
scores_bag = cross_val_score(BaggingClassifier(DecisionTreeClassifier(),
                                                n_estimators=100, random_state=42),
                              X_bc, y_bc, cv=5)
scores_rf = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42),
                             X_bc, y_bc, cv=5)
print(f'\nBreast Cancer 5-fold CV:')
print(f'  Bagging : {scores_bag.mean():.4f} ± {scores_bag.std():.4f}')
print(f'  RF      : {scores_rf.mean():.4f} ± {scores_rf.std():.4f}')
```

**출력 예시**:
```
Total features: 20
       m |   CV acc |    std
------------------------------
       1 |  0.8590 | 0.0213
       2 |  0.8940 | 0.0187
4 (√p)   |  0.9020 | 0.0124
       8 |  0.8970 | 0.0142
      15 |  0.8910 | 0.0154
      20 |  0.8850 | 0.0168

트리 쌍 간 예측 일치율:
  Bagging : 0.8732
  RF      : 0.7891  (낮음 — 다양성 증가)

직접 구현 RF accuracy : 1.0000
sklearn RF accuracy   : 1.0000

Breast Cancer 5-fold CV:
  Bagging : 0.9559 ± 0.0154
  RF      : 0.9613 ± 0.0118
```

---

## 🔗 실전 활용

- **sklearn `RandomForestClassifier/Regressor`**: 표준 구현. `n_estimators=100`, `max_features='sqrt'` (classifier) 기본.
- **Extra Trees** (Extremely Randomized Trees, Geurts 2006): random feature + **random threshold**까지 — 더 강한 randomness.
- **Tabular Kaggle baseline**: RF 먼저 시도 → GBM (XGBoost/LightGBM)과 비교.
- **Permutation importance**: RF에서 자연스럽게 추출 (Ch4-05).
- **Quantile RF** (Meinshausen 2006): 회귀에서 conditional quantile 추정 — 불확실성 정량화.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Bias 감소 불가 | 단일 tree의 bias를 이김. Boosting과 상보 |
| $n_{\text{estimators}}$ 큰 모델 | 예측이 $B$에 비례해 느림 — 실시간 prediction에 부적합할 수 있음 |
| 외삽 불가 | 각 tree의 leaf 값 기반 — extrapolation 안 됨 |
| High-dim에서 약함 | $p \gg n$에서 약함 — Lasso + FS 권장 |
| Categorical feature (many levels) | sklearn은 one-hot 필요 — ordinal 인코딩 문제 |

---

## 📌 핵심 정리

$$\boxed{\text{RF: Bagging + 각 split에서 random } m \text{ feature subset}; \ m = \sqrt{p} (\text{cls}),\ p/3 (\text{reg})}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **추가 randomness** | 각 split에서 random $m$개 feature만 후보 |
| **$\rho$ 감소** | 트리들이 다른 structure → variance 추가 감소 |
| **$\sigma^2$ 증가** | 개별 tree가 약간 더 약함 — trade-off |
| **최적 $m$** | classification $\sqrt{p}$, regression $p/3$ |
| **Breiman 정리** | generalization bound는 strength와 correlation 동시 고려 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $p = 100$, classification에서 권장 $m = ?$. $p = 1000$은?

<details>
<summary>힌트 및 해설</summary>

$p = 100$: $m = \sqrt{100} = 10$.

$p = 1000$: $m = \sqrt{1000} \approx 32$.

$m/p$의 비율: $p = 100$에서 10%, $p = 1000$에서 3.2%. 고차원일수록 더 작은 비율. — 더 많은 feature에서 더 공격적으로 subsample.

</details>

**문제 2** (심화): 회귀에서 $m = p/3$인 이유를 경험적/이론적으로 설명하라 (classification의 $\sqrt{p}$와 왜 다른가).

<details>
<summary>힌트 및 해설</summary>

Classification은 단일 split의 information value가 regression보다 큼 — 한 번의 "좋은 split"만으로도 label 분포를 크게 바꿀 수 있음. 따라서 더 공격적 subsampling ($\sqrt{p}$)이 개별 tree 성능 많이 해치지 않음.

Regression은 연속 output → 각 split이 MSE를 조금씩만 줄임. 더 많은 feature 후보 ($p/3$)가 필요 — 그래야 각 트리가 효과적인 분할을 찾음.

경험적: Breiman (2001) + 후속 hyperparameter 연구에서 관찰된 값. 정확한 이론적 유도는 아직 미해결 문제.

</details>

**문제 3** (ML 연결): NN에 "RF-style randomness"를 추가한다면 어떤 기법이 대응하는가?

<details>
<summary>힌트 및 해설</summary>

NN의 **Dropout**이 가장 유사 — 각 forward pass에서 random neuron subset 활성. "특정 sub-network가 각 예제에서만 작동" = "각 split에서만 random feature subset 고려"와 같은 정신.

**구조적 비교**:
- RF: 각 split마다 random feature (공간적 randomness)
- Dropout: 각 batch마다 random neuron (시간적 randomness)

**DropConnect** (Wan 2013): weight 자체를 dropout → 더 강한 randomness.

**Stochastic Depth** (Huang 2016): 각 forward pass에서 random layer 건너뜀 → RF의 각 tree가 다른 depth 같은 정신.

**결론**: Deep learning의 regularization 기법은 대부분 RF의 "diversity 유도 → ensemble-like effect"를 재발명한 것.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Bagging의 분산 감소](./02-bagging-variance-reduction.md) | [📚 README](../README.md) | [04. Random Forest의 수렴성 ▶](./04-rf-convergence.md) |

</div>
