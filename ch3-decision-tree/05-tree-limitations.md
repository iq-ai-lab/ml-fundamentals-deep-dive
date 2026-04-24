# 05. 결정트리의 한계 — 불안정성과 축정렬 편향

## 🎯 핵심 질문

- 결정트리가 **high variance estimator**인 이유는? 훈련 샘플 하나만 바꿔도 트리가 완전히 달라지는 현상을 어떻게 설명하는가?
- **축-정렬 분할(axis-aligned splits)**의 기하학적 한계 — 대각선 경계를 왜 잘 표현 못하는가?
- 이 두 한계가 어떻게 **앙상블의 동기**가 되는가? Bagging의 variance 감소와 RF의 추가 randomness의 필연성.
- **Oblique tree** (사선 분할)와 **HHCART**는 축정렬 편향을 어떻게 해결하는가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

본 문서는 결정트리의 **구조적 약점**을 정면으로 드러내고 — (a) 왜 sklearn 단일 트리가 실무에서 rarely best model, (b) 왜 **Random Forest가 CART를 수십 배 개선**할 수 있는지, (c) 왜 NN이 트리와 다른 strength를 가지는지에 대한 answer. 이 한계를 정확히 이해하면 **다음 챕터들의 앙상블이 왜 필요한지**가 자명해진다. "트리는 weak learner지만 모아지면 강해진다"의 수학적 메커니즘의 시작점.

---

## 📐 수학적 선행 조건

- 결정트리의 분할 알고리즘 (Ch3-01~03)
- Bias-Variance decomposition (Ch1-06)
- 가지치기 (Ch3-04)

---

## 📖 직관적 이해

### 불안정성 (high variance)

탐욕 알고리즘: 각 단계에서 현재 최적을 선택. 한 데이터 점이 바뀌면 → 어떤 feature의 IG가 미묘하게 바뀜 → 다른 feature가 선택될 수도 있음 → **그 이후 전체 subtree 구조 완전 변경**.

이는 **chaotic behavior** — 작은 input 변화가 큰 output 변화. Bias-Variance 관점에서 **매우 높은 variance**.

### 축-정렬 분할의 한계

CART는 각 분할이 $X_j \leq t$ 형태 — **단일 feature에 대한 수평·수직 선**. 대각선 경계 $X_1 + X_2 \leq t$가 진짜 decision boundary이면:

- 1개 분할로 표현 불가 → 여러 stair-step 분할로 근사 → leaf 수 폭발.
- 근사하는 step의 수 증가 → variance 증가 → overfit.

### 앙상블의 동기

두 문제를 한 번에 푸는 법: **여러 트리의 평균**.

- **Bagging**: variance 감소 (평균의 분산 법칙, Ch4-02).
- **Random Forest**: 추가 randomness → 트리 간 **correlation 감소** → 분산 추가 감소.
- **Oblique ensembles**: 각 트리에 random linear combination feature — 축정렬 편향 우회.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Instability

Estimator $\hat{f}$가 훈련 데이터 $\mathcal{D}$에 대해 **unstable**이라 함은 $\mathcal{D}$의 작은 변화(예: 한 점 perturbation)가 $\hat{f}(x_0)$에 큰 변화를 줄 때를 말한다. 정량적으로

$$\text{Instab}(\hat{f}; x_0) := \text{Var}_{\mathcal{D} \sim P}[\hat{f}(x_0)]$$

가 크다.

### 정의 5.2 — Axis-Aligned vs Oblique Split

**Axis-aligned**: 분할 $X_j \leq t$, 단일 feature.

**Oblique**: 분할 $w^\top x \leq t$, $w \in \mathbb{R}^p$ — 임의의 선형결합.

---

## 🔬 정리와 증명

### 정리 5.1 — 결정트리의 Variance는 Bagging 없이 $O(1)$ (수렴 안 함)

**명제** (informal): Full tree (완전 자람)의 variance는 $n \to \infty$에서 감소하지 않고 $O(1)$ 유지. 즉 **inconsistent**.

**증명 스케치**: 각 leaf가 단 몇 개의 점만 포함 → leaf의 예측은 거의 random. Leaf의 개수 자체가 $n$에 비례 → 평균적 variance 유지.

가지치기 + 적절한 depth로는 consistent 가능하지만 깊은 트리는 항상 high variance. $\square$

> 💡 **Bagging 후**: $B$개 독립 트리 평균의 variance는 $O(1/B)$ → variance가 체계적으로 감소. Ch4-02.

### 정리 5.2 — 축-정렬 트리의 대각선 경계 근사 비용

**명제**: 진짜 decision boundary가 $x_1 + x_2 = 0$ (대각선)일 때 CART로 $\epsilon$-정확도로 근사하려면 **$\Omega(1/\epsilon)$ 개의 leaf** 필요.

**증명 스케치**: 축정렬 분할은 stair-step. 대각선 $\{(x_1, x_2) : x_1 + x_2 = 0\}$과 stair-step의 $L_\infty$ 거리는 step 크기에 비례. $\epsilon$ 정확도에 step 크기 $\leq \epsilon$ → step 수 $\geq 1/\epsilon$ → leaf 수 $\gtrsim 1/\epsilon$. $\square$

> 📌 **대조**: Oblique tree는 1개 분할로 대각선 표현 가능. **선형 모델 + 비선형 부분 조합**이 이 문제를 근본적으로 해결.

### 정리 5.3 — Bagging으로 Variance 감소 (미리보기)

**명제**: $B$개 i.i.d. (사실은 bootstrap으로 mildly correlated) 트리의 평균 $\bar{f} = \frac{1}{B}\sum f_b$에 대해 (Ch4-02에서 자세히):

$$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2.$$

$B \to \infty$에서 $\text{Var}(\bar{f}) \to \rho \sigma^2$ — $\rho$가 낮을수록 variance 감소 더 큼.

### 정리 5.4 — Feature 선택의 Discontinuity

**명제**: Tree의 split 선택은 **argmax**를 사용 — 작은 IG 차이도 큰 결정 변화. 따라서 훈련 데이터 perturbation에 비연속.

**증명**: 두 feature A, B가 비슷한 IG. Data 작은 변화로 argmax가 A→B로 바뀜. 이후 subtree 완전히 다름. 비연속성은 argmax 함수의 표준 성질. $\square$

> 💡 **신경망과의 대조**: NN의 gradient-based learning은 연속 — 작은 input 변화 → 작은 weight 변화 → 작은 output 변화. 안정적.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.datasets import make_classification, make_moons

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. 불안정성 시연 — 데이터 1개 빼고 트리 비교
# ─────────────────────────────────────────────
X, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                            n_redundant=0, random_state=42)

def tree_str(clf):
    """간단한 트리 구조 문자열"""
    t = clf.tree_
    return [t.feature[i] for i in range(t.node_count) if t.feature[i] >= 0]

clf_full = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X, y)
structure_full = tree_str(clf_full)
print(f'Full 데이터 트리 feature split 순서: {structure_full}')

# 1개 제거하고 재학습 (여러 번)
diffs = 0
for i in range(20):
    idx = np.setdiff1d(np.arange(len(X)), [i])
    clf_perturb = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X[idx], y[idx])
    struct_p = tree_str(clf_perturb)
    if struct_p != structure_full:
        diffs += 1
print(f'20개 샘플에 대해 1점 제거했을 때 트리 구조 변경 횟수: {diffs}/20')

# ─────────────────────────────────────────────
# 2. 축정렬 편향 — make_moons 대각선 boundary
# ─────────────────────────────────────────────
X_moon, y_moon = make_moons(n_samples=200, noise=0.1, random_state=0)

for d in [3, 5, 10, 20]:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_moon, y_moon)
    n_leaves = clf.tree_.n_leaves
    acc = clf.score(X_moon, y_moon)
    print(f'Depth {d:>2}: leaves = {n_leaves:>3},  train acc = {acc:.4f}')

# ─────────────────────────────────────────────
# 3. Bagging으로 variance 감소 (정리 5.3)
# ─────────────────────────────────────────────
from sklearn.model_selection import cross_val_score

X_syn, y_syn = make_classification(n_samples=500, n_features=10, random_state=42)

models = {
    'Single Tree (depth=None)': DecisionTreeClassifier(random_state=42),
    'Single Tree (depth=5)'   : DecisionTreeClassifier(max_depth=5, random_state=42),
    'Bagging (100 trees)'     : BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=100, random_state=42),
    'Random Forest (100)'     : RandomForestClassifier(n_estimators=100, random_state=42),
}

print(f'\n{"모델":>28s} | {"5-fold accuracy (mean ± std)":>30s}')
print('-' * 62)
for name, clf in models.items():
    scores = cross_val_score(clf, X_syn, y_syn, cv=5)
    print(f'{name:>28s} | {scores.mean():.4f} ± {scores.std():.4f}')

# ─────────────────────────────────────────────
# 4. 대각선 경계 문제 시각화 데이터
# ─────────────────────────────────────────────
# 완벽한 대각선 경계 데이터
n = 500
X_diag = rng.uniform(-2, 2, (n, 2))
y_diag = (X_diag[:, 0] + X_diag[:, 1] > 0).astype(int)   # 대각선 x+y=0

# 단일 트리 (다양한 depth)
print(f'\n대각선 경계 (x_1 + x_2 = 0):')
for d in [1, 3, 5, 10]:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_diag, y_diag)
    train_acc = clf.score(X_diag, y_diag)
    print(f'  Depth {d:>2}: train acc = {train_acc:.4f}, leaves = {clf.tree_.n_leaves}')

# 사선 feature 만들어주면 해결
X_diag_aug = np.column_stack([X_diag, X_diag[:, 0] + X_diag[:, 1]])
clf_aug = DecisionTreeClassifier(max_depth=1).fit(X_diag_aug, y_diag)
print(f'  Depth 1, oblique feature 추가: train acc = {clf_aug.score(X_diag_aug, y_diag):.4f}')
print(f'  (1개 분할로 해결 — 선형 combo feature를 주면 대각선을 "축정렬"로 만듦)')

# ─────────────────────────────────────────────
# 5. RF가 대각선 경계에서 왜 더 잘 하는가
# ─────────────────────────────────────────────
print(f'\n대각선 경계 데이터에서 RF vs 단일 트리 (depth 제한 없이):')
clf_rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_diag, y_diag)
print(f'  Single Tree (full): train acc = {DecisionTreeClassifier(random_state=0).fit(X_diag, y_diag).score(X_diag, y_diag):.4f}')
print(f'  Random Forest     : train acc = {clf_rf.score(X_diag, y_diag):.4f}')
print(f'  (RF의 각 트리 예측 평균이 stair-step들의 평균 → 부드러운 대각선)')
```

**출력 예시**:
```
Full 데이터 트리 feature split 순서: [2, 1, 4, 0, 3, 2, 1]
20개 샘플에 대해 1점 제거했을 때 트리 구조 변경 횟수: 8/20

Depth  3: leaves =   8,  train acc = 0.8900
Depth  5: leaves =  26,  train acc = 0.9800
Depth 10: leaves =  80,  train acc = 1.0000
Depth 20: leaves =  80,  train acc = 1.0000

                      모델 |   5-fold accuracy (mean ± std)
--------------------------------------------------------------
    Single Tree (depth=None) | 0.8520 ± 0.0231
       Single Tree (depth=5) | 0.8520 ± 0.0182
         Bagging (100 trees) | 0.8860 ± 0.0134
        Random Forest (100)  | 0.8920 ± 0.0135

대각선 경계 (x_1 + x_2 = 0):
  Depth  1: train acc = 0.6020, leaves = 2
  Depth  3: train acc = 0.9000, leaves = 8
  Depth  5: train acc = 0.9700, leaves = 32
  Depth 10: train acc = 0.9980, leaves = 132
  Depth  1, oblique feature 추가: train acc = 1.0000
  (1개 분할로 해결 — 선형 combo feature를 주면 대각선을 "축정렬"로 만듦)

대각선 경계 데이터에서 RF vs 단일 트리 (depth 제한 없이):
  Single Tree (full): train acc = 1.0000
  Random Forest     : train acc = 0.9960
  (RF의 각 트리 예측 평균이 stair-step들의 평균 → 부드러운 대각선)
```

---

## 🔗 실전 활용

- **단일 트리 ≠ 표준**: 해석성이 primary 목적이 아니면 단일 트리 대신 RF/GBM 사용.
- **Feature engineering**: 대각선 경계가 예상되면 `X_1 + X_2`, `X_1 - X_2` 같은 composite feature 추가.
- **Oblique trees**: **HHCART** (House-Holder CART, Wickramarachchi 2015), **Oblique RF** (Menze 2011) — 각 split에 linear combo.
- **Deep Learning comparison**: NN은 learned feature로 대각선을 자동 표현 — 트리의 축정렬 약점 우회.
- **Tabular data에서 RF/GBM이 여전히 강한 이유**: axis-aligned split이 범주형 + 희소한 interaction에 적합.

---

## ⚖️ 가정과 한계

| 한계 | 설명 | 해결 |
|------|------|------|
| High variance | 탐욕 + argmax → chaotic | Bagging, RF, GBM |
| 축정렬 편향 | 대각 경계 표현 비효율 | Oblique split, feature engineering |
| 외삽 불가 | Leaf 값 고정 | 선형 모델 + 트리 혼합 (M5) |
| 불연속 | Step function | Kernel smoothing after tree |

---

## 📌 핵심 정리

$$\boxed{\text{Tree의 두 약점 → 앙상블의 두 수단:}\quad \text{High variance}\xrightarrow{\text{Bagging}}\text{평균},\ \text{축정렬}\xrightarrow{\text{RF}}\text{randomness}}$$

| 약점 | 원인 | 앙상블의 해결 |
|------|------|------|
| **불안정성** | Greedy + argmax의 비연속성 | Bagging의 averaging → variance 감소 |
| **축정렬 편향** | $X_j \leq t$ 형태 제약 | Feature subsampling으로 ρ 감소, 평균의 부드러움 |
| **Overfit** | Full tree의 제로 train error | Pre/post-pruning, 앙상블의 regularization |
| **외삽 불가** | Leaf 값 고정 | 결정적 — 앙상블로도 해결 안 됨 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 대각선 boundary에서 depth $d$ 트리의 leaf 수 하한을 추정하라 (어떤 $\epsilon$으로 근사할 때).

<details>
<summary>힌트 및 해설</summary>

대각선을 $1/\epsilon$ 개의 stair-step으로 근사 → 최소 $1/\epsilon$개 leaf. 각 leaf가 $\epsilon \times \epsilon$ 크기 region.

Depth $d$ binary tree의 최대 leaf 수 = $2^d$. 따라서 $2^d \geq 1/\epsilon$ → $d \geq \log_2(1/\epsilon)$.

→ **지수적 깊이 필요**. $\epsilon = 0.01$이면 depth ≥ 7. $\epsilon = 0.001$이면 depth ≥ 10.

</details>

**문제 2** (심화): Bagging이 **bias**를 감소시키지 않는 이유를 평균의 선형성으로 설명하라. 왜 Boosting은 bias를 감소시키는가?

<details>
<summary>힌트 및 해설</summary>

**Bagging**: $\bar{f}(x) = \frac{1}{B}\sum f_b(x)$. $\mathbb{E}[\bar{f}(x)] = \mathbb{E}[f_b(x)]$ (i.i.d. 가정) — bagging의 평균이 single model의 평균과 같음 → **같은 bias**. Bagging은 **variance만 감소**.

**Boosting**: 순차적으로 **이전 모델의 오류를 보정**하는 방향으로 새 모델 추가. $F_t = F_{t-1} + h_t$, $h_t$는 잔차/negative gradient에 fit. 각 step이 bias를 줄이도록 설계. Ch5-03에서 함수공간 경사하강법으로.

**결론**: Bagging ↔ variance, Boosting ↔ bias. **상보적**이고 실무에서 자주 같이 사용 (XGBoost가 base learner로 shallow tree = low-variance learner).

</details>

**문제 3** (ML 연결): NN은 **learned features**를 씀 → 축정렬 편향이 없음. 그럼에도 tabular 데이터에서 RF/GBM이 NN보다 자주 이기는 이유는?

<details>
<summary>힌트 및 해설</summary>

Tabular 데이터 특징: (a) **heterogeneous features** — 숫자·범주·binary 혼재, (b) **희소한 interaction** — 대부분 feature pair는 결합 effect 없음, (c) **작은 $n$** — overfit 위험.

Tree-based 모델의 이점:
- **자동 missing handling**: surrogate split.
- **mixed feature type** — scale-invariant.
- **Interaction은 필요한 곳에만** 자동 발견 — greedy하게.
- **적은 hyperparameter tuning** — OOTB 잘 작동.

NN의 약점 (tabular):
- 표준 architecture (MLP) — feature interaction을 자동 발견 못함.
- scale-sensitive, 전처리 필요.
- 많은 데이터 필요.

**결론**: 대각선 편향은 **fundamental**하지만 tabular의 다른 특성들이 이점을 상쇄. Image/text처럼 **pixel/word이 homogeneous한 도메인**에서 NN이 기하학적 편향을 정확히 활용 (CNN의 translation invariance 등).

→ "결정트리는 한계가 있지만 tabular의 sweet spot에서 여전히 강함" — **No Free Lunch**의 한 사례.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. 가지치기](./04-pruning-cost-complexity.md) | [📚 README](../README.md) | [Ch4-01. Bootstrap과 OOB Error ▶](../ch4-bagging-rf/01-bootstrap-oob.md) |

</div>
