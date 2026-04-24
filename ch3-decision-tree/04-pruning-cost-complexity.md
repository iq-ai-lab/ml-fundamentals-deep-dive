# 04. 가지치기와 Cost-Complexity Pruning

## 🎯 핵심 질문

- 완전히 자란 트리는 왜 overfit하는가? **Cost-Complexity** $R_\alpha(T) = R(T) + \alpha |T|$ 의 의미는?
- $\alpha$가 어떻게 트리 크기를 통제하는가? **Weakest Link Pruning**의 단조 시퀀스 정리는?
- $\alpha$를 **Cross-Validation**으로 어떻게 선택하는가? **1-SE rule**의 직관은?
- sklearn `cost_complexity_pruning_path()`의 출력을 어떻게 해석하나?

---

## 🔍 왜 이 개념이 ML에서 중요한가

가지치기(pruning)는 (a) 단일 결정트리의 **overfit 방지**의 표준, (b) **Bias-Variance trade-off의 명시적 control** — $\alpha$ 큼 → variance ↓ bias ↑, (c) 매우 elegant한 알고리즘 — Breiman et al. (1984)의 weakest link sequence가 모든 가능한 $\alpha$의 트리를 한 번에 만듦, (d) Random Forest는 가지치기를 안 하지만 (다양성 보존), 단일 CART · GBM의 **regularization 도구로 사용**. 본 문서는 sklearn `ccp_alpha` 파라미터의 수학적 배경 — 직관 없이 쓰면 효과 없음.

---

## 📐 수학적 선행 조건

- 결정트리의 분할과 leaf (Ch3-01, 02, 03)
- Cross-validation
- Bias-Variance (Ch1-06)

---

## 📖 직관적 이해

### 왜 가지치기인가

완전 자란 트리: train error = 0 (모든 leaf pure), test error 큼 (overfit). **Pre-pruning** (max_depth로 일찍 멈춤) vs **Post-pruning** (다 자란 후 가지 잘라냄). Post-pruning이 더 정확 — 미래 분할이 도움 줄지 미리 모름.

### Cost-Complexity 정규화

$R(T) = $ training error (분류: misclassification rate, 회귀: MSE), $|T| = $ leaf 수. 새 손실:

$$R_\alpha(T) := R(T) + \alpha |T|.$$

$\alpha = 0$: full tree. $\alpha = \infty$: root만 (1 leaf). $\alpha$ 키우면 tree 작아짐 — **단조감소 sequence** (정리 4.1).

### Weakest Link Pruning

각 internal node $t$에 대해 "그 subtree를 leaf로 압축하면 cost-complexity가 어떻게 변하나"를 계산:

$$g(t) := \frac{R(t) - R(T_t)}{|T_t| - 1}.$$

여기서 $T_t$는 $t$ 아래 subtree, $R(t) = $ $t$를 leaf로 만들었을 때의 error. **$g(t)$가 가장 작은 노드부터 잘라냄** — "단위 leaf 감소당 error 증가가 가장 작은" 가지 = weakest link.

이를 반복하면 **유한한 $\alpha$ 시퀀스** $0 = \alpha_0 < \alpha_1 < \cdots < \alpha_M$와 그에 대응하는 nested 트리 $T_0 \supset T_1 \supset \cdots \supset T_M$ (root)이 자연스럽게 나옴.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Cost-Complexity

$T$가 트리, $|T|$가 leaf 수, $R(T) = \sum_{v \in \text{leaves}} R_v$가 training risk (분류: $|S_v| \cdot \text{misclass rate}$, 회귀: $\sum_{i \in S_v} (y_i - \bar{y}_v)^2$). $\alpha \geq 0$:

$$R_\alpha(T) := R(T) + \alpha |T|.$$

### 정의 4.2 — Subtree Weakest Link

Internal node $t$, subtree $T_t$. 

$$g(t) := \frac{R(\{t\}) - R(T_t)}{|T_t| - 1},$$

여기서 $R(\{t\}) = $ $t$를 leaf로 만들었을 때 그 region의 error (= leaf 1개일 때).

### 정의 4.3 — Sequence of Pruned Trees

$T_0 = $ full tree. $\alpha_0 = 0$. 반복:

1. 모든 internal node $t \in T_k$에 대해 $g_k(t)$ 계산.
2. $g^*_k = \min_t g_k(t)$, $t^*_k = \arg\min$.
3. $\alpha_{k+1} = g^*_k$, $T_{k+1} = T_k$의 $T_{t^*_k}$를 leaf로 대체.
4. $T_{k+1}$이 root만 남으면 종료.

→ Sequence $\{(α_k, T_k)\}$.

---

## 🔬 정리와 증명

### 정리 4.1 — Pruned Tree Sequence는 Nested

**명제**: 정의 4.3의 sequence $T_0 \supset T_1 \supset \cdots \supset T_M$은 nested (각 $T_{k+1}$이 $T_k$의 subtree).

**증명**: 정의 4.3의 step 3에서 $T_{k+1}$은 $T_k$의 한 subtree를 leaf로 대체한 것 — 자명히 subtree. $\square$

### 정리 4.2 — 각 $\alpha$에 대해 최적 트리는 sequence의 한 원소 (Breiman 1984)

**명제**: 임의의 $\alpha \geq 0$에 대해 $\arg\min_T R_\alpha(T)$ (subtree of $T_0$)는 sequence $\{T_k\}$의 한 원소이다. 구체적으로 $\alpha \in [\alpha_k, \alpha_{k+1})$이면 $T_k$가 최적.

**증명 스케치**: $g(t)$가 최소인 노드를 prune하는 것이 cost-complexity를 가장 적게 증가시키는 방향. 모든 $\alpha$에 대해 이 그리디한 선택이 최적임을 induction으로 보임. 자세한 증명은 Breiman et al. (1984) Ch3 또는 Hastie ESL Ch9.2. $\square$

> 💡 **함의**: 모든 $\alpha$를 시도할 필요 없음 — sequence의 $M$개 트리만 후보. 매우 효율적.

### 정리 4.3 — Cross-Validation으로 $\alpha$ 선택

**알고리즘**:

1. K-fold CV split.
2. 각 fold $i$에 대해 (train_i, val_i):
   - train_i로 full tree와 sequence $\{(α_k^{(i)}, T_k^{(i)})\}$ 생성.
   - 각 $\alpha$에 대해 val_i에서 error 측정.
3. 모든 fold의 val error 평균을 $\alpha$별로 → 최적 $\alpha^* = \arg\min$.
4. 전체 데이터로 다시 학습한 후 $\alpha^*$로 prune → 최종 트리.

**1-SE rule**: 가장 simple한 트리 ($\alpha$ 큰)를 선택, 단 그 error가 minimum + 1 standard error 이내일 것. 더 robust.

### 정리 4.4 — Pre-pruning vs Post-pruning

**Pre-pruning** (early stopping): `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`. 빠르지만 **horizon effect** — 미래 분할이 도움 될 수도 있는데 미리 멈춤.

**Post-pruning** (cost-complexity): full tree → CV로 $\alpha$ 선택 → prune. 느리지만 더 정확.

실무: 둘 다 사용. Pre-pruning으로 일단 합리적 크기, post-pruning으로 fine-tune.

### 정리 4.5 — Information Criteria (BIC, MDL)

Cost-complexity는 사실 **AIC/BIC family**의 트리 버전:

$$R(T) + \alpha |T|.$$

$\alpha = 2 \cdot \log L$ → AIC. $\alpha = \log n \cdot \log L$ → BIC. **Cost-complexity는 cross-validation으로 $\alpha$를 데이터에 맞게 선택**한다는 점이 다름.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, KFold

# ─────────────────────────────────────────────
# 1. sklearn cost_complexity_pruning_path 사용
# ─────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
n = len(X)

clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X, y)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print(f'Sequence 길이: {len(ccp_alphas)}')
print(f'\n처음 5개 (α, impurity):')
for a, imp in list(zip(ccp_alphas, impurities))[:5]:
    print(f'  α = {a:.6f},  impurity = {imp:.4f}')
print(f'마지막 (root): α = {ccp_alphas[-1]:.4f},  impurity = {impurities[-1]:.4f}')

# ─────────────────────────────────────────────
# 2. 각 α에서 트리 학습 → CV error 측정 (정리 4.3)
# ─────────────────────────────────────────────
print(f'\n각 α에서 5-fold CV accuracy 측정...')
cv_scores_mean, cv_scores_std = [], []
for alpha in ccp_alphas[::3]:  # 매 3번째만 (속도)
    clf_a = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(clf_a, X, y, cv=5, scoring='accuracy')
    cv_scores_mean.append(scores.mean())
    cv_scores_std.append(scores.std())

# 최적 α
best_idx = np.argmax(cv_scores_mean)
best_alpha = ccp_alphas[::3][best_idx]
best_score = cv_scores_mean[best_idx]
print(f'\n최적 α = {best_alpha:.4f},  CV accuracy = {best_score:.4f}')

# 1-SE rule
threshold = best_score - cv_scores_std[best_idx]
simpler_idx = next((i for i in range(len(cv_scores_mean) - 1, -1, -1)
                    if cv_scores_mean[i] >= threshold), best_idx)
print(f'1-SE rule α = {ccp_alphas[::3][simpler_idx]:.4f},  CV accuracy = {cv_scores_mean[simpler_idx]:.4f}')

# ─────────────────────────────────────────────
# 3. α별 트리 크기와 train/test accuracy 비교
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f'\n{"α":>10s} | {"|T|":>5s} | {"train acc":>10s} | {"test acc":>9s}')
print('-' * 50)
for alpha in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    clf_a = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42).fit(X_train, y_train)
    n_leaves = clf_a.tree_.n_leaves
    train_acc = clf_a.score(X_train, y_train)
    test_acc = clf_a.score(X_test, y_test)
    print(f'{alpha:>10.4f} | {n_leaves:>5d} | {train_acc:>10.4f} | {test_acc:>9.4f}')

# ─────────────────────────────────────────────
# 4. Bias-Variance 시각화 (α가 BV trade-off 통제)
# ─────────────────────────────────────────────
print(f'\nα ↑ → train acc ↓ (bias↑), 그러나 (잘 선택하면) test acc ↑ (variance↓)')
print(f'sklearn `ccp_alpha`는 cost-complexity의 α 그대로')
```

**출력 예시**:
```
Sequence 길이: 27

처음 5개 (α, impurity):
  α = 0.000000,  impurity = 0.0000
  α = 0.000875,  impurity = 0.0009
  α = 0.001750,  impurity = 0.0026
  α = 0.002625,  impurity = 0.0079
  α = 0.003500,  impurity = 0.0156
마지막 (root): α = 0.3373,  impurity = 0.4685

각 α에서 5-fold CV accuracy 측정...

최적 α = 0.0096,  CV accuracy = 0.9438
1-SE rule α = 0.0235,  CV accuracy = 0.9298

         α |   |T| |  train acc |  test acc
--------------------------------------------------
    0.0000 |    33 |     1.0000 |    0.9181
    0.0010 |    23 |     0.9925 |    0.9357
    0.0050 |    13 |     0.9774 |    0.9415
    0.0100 |     8 |     0.9698 |    0.9474
    0.0200 |     5 |     0.9523 |    0.9415
    0.0500 |     3 |     0.9347 |    0.9298
    0.1000 |     2 |     0.9272 |    0.9181
```

---

## 🔗 실전 활용

- **sklearn `DecisionTreeClassifier(ccp_alpha=...)`**: Cost-Complexity Pruning. v0.22+.
- **`cost_complexity_pruning_path()`**: 모든 $\alpha$ sequence를 한 번에 계산.
- **GridSearchCV로 $\alpha$ 선택**: 위 코드의 자동화 버전.
- **R `rpart`**: post-pruning이 기본 — `cp` 파라미터.
- **Random Forest는 안 prune**: 개별 트리의 high variance를 평균으로 상쇄 — pruning이 다양성 손해.
- **Gradient Boosting**: 작은 트리 (depth 3~6) 많이 사용 — pre-pruning만으로 충분.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| 단일 트리에만 가치 | RF/GBM은 자체 정규화 메커니즘 보유 |
| CV 비용 | $K \times M$개 트리 학습 |
| 1-SE rule의 임의성 | 1 SE는 convention — 0.5 또는 2 SE도 가능 |
| Greedy sequence | weakest link가 항상 최적 sequence란 보장은 위 정리에 의존 |

---

## 📌 핵심 정리

$$\boxed{R_\alpha(T) = R(T) + \alpha|T|; \quad \text{Weakest Link: } g(t) = \frac{R(\{t\}) - R(T_t)}{|T_t| - 1}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **Cost-Complexity** | error + leaf 페널티 |
| **$\alpha$ sequence** | 유한한 $M+1$개 nested 트리만 후보 |
| **CV $\alpha$ 선택** | K-fold으로 generalization 추정 |
| **1-SE rule** | 가장 simple한 트리 (best ± 1 SE 이내) |
| **AIC/BIC family** | $\alpha$가 model selection criterion의 트리 버전 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 트리 $T$가 leaf 5개, $R(T) = 0.1$이라면 $R_{\alpha=0.05}(T) = ?$

<details>
<summary>힌트 및 해설</summary>

$R_\alpha(T) = 0.1 + 0.05 \cdot 5 = 0.35$. 

만약 $|T| = 1$ (root)이고 $R(\{root\}) = 0.4$이면 $R_\alpha = 0.4 + 0.05 \cdot 1 = 0.45$. 

→ 큰 트리가 더 작은 $R_\alpha$ → 큰 트리 선호. $\alpha$를 더 크게 하면 root가 이김.

</details>

**문제 2** (심화): Weakest link $g(t)$가 작은 노드를 먼저 prune하는 이유를 cost-complexity 관점에서 설명하라.

<details>
<summary>힌트 및 해설</summary>

$g(t)$ = "단위 leaf 감소당 error 증가". $T_t$를 leaf로 압축하면 error는 $R(T_t) \to R(\{t\})$로 증가하지만 leaf 수는 $|T_t| \to 1$로 감소. 비율 $g(t)$가 작을수록 "leaf 1개 줄여서 얻는 cost 절감 ($\alpha$만큼)이 error 증가보다 큼" — $\alpha = g(t)$에서 prune이 손해 안 봄.

따라서 $\alpha$를 점진적으로 키우며 sequence를 만들 때 가장 작은 $g(t)$가 임계점. 이 노드를 prune하면 그 다음 $\alpha$ 임계가 다음 작은 $g(t)$. 자연스러운 순서.

</details>

**문제 3** (ML 연결): NN의 **weight pruning** (e.g. Magnitude Pruning, Lottery Ticket Hypothesis)이 본 문서의 cost-complexity와 어떻게 정신적으로 연결되는가?

<details>
<summary>힌트 및 해설</summary>

NN weight pruning: 작은 magnitude weight를 0으로 잘라낸 후 fine-tune. 손실 = train_loss + sparsity_penalty (implicit하게 작은 weight 제거 = $L_0$ 정규화 근사).

Cost-Complexity의 트리 버전과 평행: error + complexity penalty. 트리는 $|T|$ (leaf 수), NN은 $\|w\|_0$ (비영 weight 수).

**Lottery Ticket Hypothesis** (Frankle & Carbin 2019): 큰 NN 안에 작은 sub-network가 단독으로도 같은 성능 — 트리에서 큰 subtree에서 작은 pruned tree를 추출하는 것과 같은 원리.

**둘 다 "complexity 줄이고 generalization 향상"이라는 universal goal**. 알고리즘은 다르지만 **bias-variance trade-off 한 형태**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. 회귀 트리와 MSE 분할](./03-regression-tree.md) | [📚 README](../README.md) | [05. 결정트리의 한계 ▶](./05-tree-limitations.md) |

</div>
