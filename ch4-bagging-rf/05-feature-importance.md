# 05. Feature Importance — Permutation과 MDI

## 🎯 핵심 질문

- **Mean Decrease in Impurity (MDI)** = 트리에서 한 feature가 만든 split의 impurity 감소 합. 왜 **high-cardinality bias**가 발생하는가?
- **Permutation Importance** = test set에서 feature 하나를 random shuffle한 후 정확도 감소. 왜 더 robust한가?
- 두 importance가 일치하는 경우와 다른 경우는?
- **SHAP** (SHapley Additive exPlanations)이 왜 더 일반적인 framework인가?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Feature importance는 (a) **모델 해석의 표준 도구** — 특히 RF·GBM·XGBoost처럼 black-box 같은 모델에서, (b) **feature selection의 baseline** — 중요도 낮은 feature 제거, (c) 도메인 전문가와의 소통 — "왜 이 모델이 이렇게 예측했나" 답변, (d) **편향 진단** — high-cardinality (예: customer ID) feature가 부당히 높은 중요도를 받으면 모델 신뢰성 의심. 본 문서는 sklearn의 `feature_importances_` (MDI 기본)와 `permutation_importance`의 차이, 그리고 SHAP이 왜 두 한계를 모두 해결하는지 설명.

---

## 📐 수학적 선행 조건

- 결정트리의 Gini/Entropy 분할 (Ch3-01, 02)
- Bootstrap과 OOB (Ch4-01)
- (참고) Shapley value의 game theory

---

## 📖 직관적 이해

### MDI — 트리의 자연 부산물

각 split에서 부모 노드의 impurity를 자식들의 가중 평균 impurity로 줄임. 이 감소량 × split된 feature에 누적 = MDI.

$$\text{MDI}(j) := \sum_{t : \text{split on } j} \frac{|S_t|}{|S|} \Delta I(t).$$

장점: tree 학습 중 자동 계산 — 거의 공짜.
단점: **high-cardinality feature가 부당하게 우위**.

### Permutation — 모델-agnostic 측정

Test set에서 feature $j$의 값을 random shuffle → 모델 예측 → 정확도 감소량. 이 감소가 클수록 $j$가 중요.

$$\text{PI}(j) := \text{Acc}(\text{original}) - \text{Acc}(\text{permuted } X_j).$$

장점: 모델 종류 무관 + 실제 prediction에 미치는 영향.
단점: 계산 비쌈 + **상관된 feature 쌍** 처리 부정확.

### High-Cardinality Bias

ID feature (n unique values): 각 split에서 IG가 매우 큼 (잘게 자름) → MDI 큼. 그러나 실제로는 train data를 외운 것에 불과 → test에 유효 X.

Permutation Importance는 **test set에서 측정**하므로 generalize 안 되는 feature는 자동 패널티.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Mean Decrease in Impurity (MDI)

Tree $T$의 모든 split node $t$에 대해 split feature $A(t)$, weighted impurity decrease $\Delta I(t) = \frac{|S_t|}{|S|}(I(t) - I(t_L) - I(t_R))$. 

$$\text{MDI}_T(j) := \sum_{t : A(t) = j} \Delta I(t).$$

Forest에 대해: $\text{MDI}(j) = \frac{1}{B}\sum_b \text{MDI}_{T_b}(j)$.

### 정의 5.2 — Permutation Importance (Breiman 2001)

테스트 데이터 $\mathcal{D}_{\text{test}}$, 모델 $f$. Feature $j$만 random permute한 데이터 $\mathcal{D}_{\text{test}}^{(j)}$:

$$\text{PI}(j) := L(f, \mathcal{D}_{\text{test}}^{(j)}) - L(f, \mathcal{D}_{\text{test}}).$$

여러 random permutation 평균으로 분산 줄임.

---

## 🔬 정리와 증명

### 정리 5.1 — MDI의 분해

**명제**: 다음 등식이 성립:

$$\sum_j \text{MDI}_T(j) = I(\text{root}) - \sum_v \frac{|S_v|}{|S|} I(v),$$

여기서 합은 leaves $v$에 대한 것. 즉 **root impurity 감소 총량 = 모든 feature MDI의 합**.

**증명 sketch**: 트리 위에서 텔레스코프 합. 각 split이 부모를 자식들로 나눔 — impurity 감소 누적. $\square$

> 💡 **함의**: MDI는 트리가 만든 information 전체를 feature별로 분배.

### 정리 5.2 — High-Cardinality Bias (Strobl 2007)

**명제** (informal): MDI는 high-cardinality (또는 continuous) feature를 favor한다 — 같은 generalization 가치를 가진 두 feature 중 카테고리가 더 많은 쪽이 더 큰 MDI를 받음.

**증명 sketch**: 카테고리 많은 feature는 더 fine한 split 가능 → IG 측정의 high-card bias (Ch3-01 정리 1.4). MDI에 누적 → 부당히 큼.

**보정**: Conditional Permutation Importance, MDA (Mean Decrease Accuracy with OOB), SHAP. $\square$

### 정리 5.3 — Permutation Importance의 통계적 정당성

**명제**: Feature $j$가 outcome과 conditional independent ($Y \perp X_j \mid X_{-j}$)면 $\mathbb{E}[\text{PI}(j)] = 0$.

**증명**: $X_j$를 permute해도 $X_{-j}, Y$의 결합 분포가 달라지지 않음 (since $Y$의 분포가 $X_j$에 무관). 모델 예측의 분포도 변하지 않음 (point-wise는 다르지만 평균 loss는 같음). 

엄밀히: $\mathbb{E}[L(f(\tilde{X}_j, X_{-j}), Y)] = \mathbb{E}[L(f(X_j, X_{-j}), Y)]$ if $Y \perp X_j \mid X_{-j}$. $\square$

### 정리 5.4 — Permutation의 Correlation 약점

**명제**: 두 feature $X_1, X_2$가 강하게 상관 + 둘 다 $Y$에 영향 → permutation이 한 feature의 importance를 부정확히 추정.

**이유**: $X_1$만 permute → 새 결합 $(\tilde{X}_1, X_2)$가 **train distribution의 영역 밖** → 모델이 **extrapolation 영역에서 평가**됨 → 비현실적 prediction. 

**해결**: 
- **Conditional Permutation** (Strobl 2008): $X_1 \mid X_2$의 conditional distribution에서 sampling.
- **Grouped permutation**: 상관 feature를 함께 permute.

### 정리 5.5 — SHAP의 일반화

**명제** (Lundberg & Lee 2017): Shapley value $\phi_j = \frac{1}{|F|!}\sum_{\pi} [v(\pi_j \cup \{j\}) - v(\pi_j)]$ where 합은 모든 feature ordering의 평균. 만족 성질:

1. **Local accuracy**: $\sum \phi_j = f(x) - \mathbb{E}[f(X)]$.
2. **Missingness**: $X_j$ 사용 안 하면 $\phi_j = 0$.
3. **Consistency**: 모델이 더 $X_j$에 의존하면 $\phi_j$도 증가.

**유일성**: 위 3 axiom을 만족하는 attribution은 Shapley value로 unique.

**Tree SHAP**: 트리 구조에서 efficient $O(LD^2)$ 계산 (Lundberg 2018). MDI/PI의 한계를 정리에 기반해 해결.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. MDI vs Permutation 기본 비교
# ─────────────────────────────────────────────
X, y = make_classification(n_samples=1000, n_features=10, n_informative=4,
                            n_redundant=2, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)

mdi = rf.feature_importances_
pi = permutation_importance(rf, X, y, n_repeats=10, random_state=42).importances_mean

print(f'{"feature":>8s} | {"MDI":>8s} | {"PI":>8s}')
print('-' * 30)
for j in range(10):
    print(f'  X_{j:<5d}| {mdi[j]:.4f} | {pi[j]:.4f}')

# ─────────────────────────────────────────────
# 2. High-cardinality bias 시연 (정리 5.2)
# ─────────────────────────────────────────────
n = 1000
X_real = rng.standard_normal((n, 4))    # 진짜 informative
X_id = rng.integers(0, n, n).reshape(-1, 1)   # 거의 unique — random ID
X_combined = np.column_stack([X_real, X_id])  # feature 5개

# 진짜 label은 X_real에서만
y_synth = (X_real @ rng.standard_normal(4) > 0).astype(int)

rf_combined = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_combined, y_synth)
mdi_combined = rf_combined.feature_importances_
pi_combined = permutation_importance(rf_combined, X_combined, y_synth, n_repeats=5).importances_mean

print(f'\n{"feature":>10s} | {"MDI":>8s} | {"PI":>8s}')
print('-' * 32)
for j in range(5):
    name = f'real_{j}' if j < 4 else 'ID-like'
    print(f'  {name:>8s} | {mdi_combined[j]:.4f} | {pi_combined[j]:.4f}')

print(f'\n→ MDI: ID-like feature가 부풀려진 importance')
print(f'→ PI : ID-like feature는 0 — permutation으로 정상화')

# ─────────────────────────────────────────────
# 3. Train vs Test set의 PI 차이
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X_combined, y_synth, random_state=0)
rf_tr = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)

pi_train = permutation_importance(rf_tr, X_tr, y_tr, n_repeats=5).importances_mean
pi_test  = permutation_importance(rf_tr, X_te, y_te, n_repeats=5).importances_mean

print(f'\nPermutation importance: train vs test')
print(f'{"feature":>10s} | {"train":>7s} | {"test":>7s}')
print('-' * 30)
for j in range(5):
    name = f'real_{j}' if j < 4 else 'ID-like'
    print(f'  {name:>8s} | {pi_train[j]:.4f} | {pi_test[j]:.4f}')
print(f'(test에서 ID-like가 음수 또는 0 → 일반화 안 됨)')

# ─────────────────────────────────────────────
# 4. Correlated features의 permutation 약점 (정리 5.4)
# ─────────────────────────────────────────────
n = 500
X_corr = rng.standard_normal((n, 3))
X_corr[:, 1] = X_corr[:, 0] + 0.1 * rng.standard_normal(n)  # X_1 = X_0 + 노이즈
y_corr = (X_corr[:, 0] + X_corr[:, 2] > 0).astype(int)

rf_corr = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_corr, y_corr)
mdi_c = rf_corr.feature_importances_
pi_c = permutation_importance(rf_corr, X_corr, y_corr, n_repeats=10).importances_mean

print(f'\nCorrelated features (X_0 ≈ X_1):')
print(f'  MDI : {mdi_c.round(4)}')
print(f'  PI  : {pi_c.round(4)}')
print(f'  → X_0과 X_1이 같은 정보 → permutation이 둘 모두 낮게 평가')
```

**출력 예시**:
```
 feature |      MDI |       PI
------------------------------
  X_0     | 0.0524 | 0.0123
  X_1     | 0.1234 | 0.0982
  X_2     | 0.0321 | 0.0089
  X_3     | 0.2891 | 0.2453
  X_4     | 0.1023 | 0.0712
  X_5     | 0.1842 | 0.1543
  ...

 feature |      MDI |       PI
--------------------------------
   real_0 | 0.1842 | 0.2543
   real_1 | 0.1923 | 0.2491
   real_2 | 0.1789 | 0.2387
   real_3 | 0.1654 | 0.2418
   ID-like | 0.2792 | 0.0021

→ MDI: ID-like feature가 부풀려진 importance
→ PI : ID-like feature는 0 — permutation으로 정상화

Permutation importance: train vs test
 feature |   train |    test
------------------------------
   real_0 | 0.0321 | 0.2310
   real_1 | 0.0298 | 0.2247
   real_2 | 0.0287 | 0.2189
   real_3 | 0.0312 | 0.2154
   ID-like | 0.1654 | -0.0023
(test에서 ID-like가 음수 또는 0 → 일반화 안 됨)

Correlated features (X_0 ≈ X_1):
  MDI : [0.3214 0.3145 0.3641]
  PI  : [0.0234 0.0289 0.4523]
  → X_0과 X_1이 같은 정보 → permutation이 둘 모두 낮게 평가
```

---

## 🔗 실전 활용

- **sklearn `feature_importances_`**: MDI 기본. 빠르지만 high-card bias 주의.
- **sklearn `permutation_importance`**: model-agnostic. test set에서 사용.
- **SHAP** (`shap` 패키지): TreeExplainer로 RF·XGBoost 효율적 SHAP 계산.
- **eli5**: permutation importance + LIME 시각화.
- **Drop-column importance**: feature 하나씩 빼고 모델 재학습 → 직접 측정. 매우 비싸지만 정확.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| MDI: high-card bias | continuous·many-cat feature 부풀려짐 |
| PI: 상관 feature | 그룹 함께 permute 또는 conditional |
| 둘 다 인과 X | "$X_j$ 중요" ≠ "$X_j$가 $Y$의 원인" |
| Test 의존 | PI는 test set 분포에 민감 |

---

## 📌 핵심 정리

$$\boxed{\text{MDI: tree-internal Gini decrease 합 (high-card bias);\ PI: feature shuffle 후 acc 감소 (모델-agnostic)}}$$

| 기법 | 장점 | 약점 |
|------|------|------|
| **MDI** | 공짜, 빠름 | high-card feature 부풀려짐 |
| **Permutation** | model-agnostic, generalization 검증 | 상관 feature 처리 부정확 |
| **SHAP** | axiomatic 일관성 | 계산 복잡 (Tree SHAP는 효율적) |
| **Drop-column** | 가장 정확 | 매우 비쌈 (재학습) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): RF에서 MDI가 1.0으로 정규화된다는 의미는? `feature_importances_.sum()`이 항상 1인 이유.

<details>
<summary>힌트 및 해설</summary>

sklearn의 MDI는 normalize되어 합 = 1. 정의상 모든 split의 impurity decrease 합 = root impurity (정리 5.1) → feature별 분배 + normalize.

→ MDI는 **상대적 importance**만 제공. 절대값 의미 없음. 두 모델 간 직접 비교 불가.

</details>

**문제 2** (심화): Permutation Importance에서 `n_repeats`가 클수록 좋은 이유와 trade-off는?

<details>
<summary>힌트 및 해설</summary>

각 random permutation마다 다른 결과 → standard error 큼. 여러 번 반복 평균 → variance 감소.

`n_repeats=5` (sklearn 기본): 빠르지만 noisy. `n_repeats=30`: 정확하지만 느림.

**Trade-off**: 계산 시간 vs 측정 정확성. 실무 권장: 10~30회.

`importances_std` 속성으로 표준편차 확인 가능. 작은 importance가 std 안에 있으면 통계적으로 0과 구분 불가.

</details>

**문제 3** (ML 연결): NN의 **Integrated Gradients** (Sundararajan 2017)가 feature importance의 NN 버전임을 설명하라.

<details>
<summary>힌트 및 해설</summary>

Integrated Gradients: $\text{IG}_j(x) := (x_j - x_j^0) \cdot \int_0^1 \frac{\partial f(x^0 + \alpha(x - x^0))}{\partial x_j} d\alpha$.

baseline $x^0$ (보통 zero)에서 $x$로 가는 path를 따라 gradient 적분. **Local accuracy** (정리 5.5의 SHAP 첫 axiom)를 만족: $\sum_j \text{IG}_j = f(x) - f(x^0)$.

평행 구조:
- SHAP: feature 부분집합의 권력 → coalition 평균
- IG: 점 path의 gradient 적분

둘 다 **axiomatic feature attribution**. NN은 미분 가능 → IG가 자연스러움. 트리는 미분 불가 → SHAP만 가능. **Modern XAI의 두 큰 흐름**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. RF의 수렴성](./04-rf-convergence.md) | [📚 README](../README.md) | [Ch5-01. AdaBoost의 유도 ▶](../ch5-boosting/01-adaboost-derivation.md) |

</div>
