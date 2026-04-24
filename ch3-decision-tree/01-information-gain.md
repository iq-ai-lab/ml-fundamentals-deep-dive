# 01. 결정트리의 분할 기준 — 정보이득

## 🎯 핵심 질문

- Information Gain $IG(S, A) = H(S) - \sum_v \frac{|S_v|}{|S|} H(S_v)$가 왜 정확히 **상호정보량 $I(Y; A)$와 동치**인가?
- 엔트로피 $H(Y) = -\sum p_k \log p_k$는 어떤 의미에서 "불확실성"을 재는가?
- ID3 알고리즘은 어떻게 IG를 탐욕적으로 사용해 트리를 자라게 하는가?
- IG의 한계 — **high-cardinality feature 편향** ($\log_2 n$ 카테고리 feature를 선호)와 그 해결 (Gain Ratio, Gini)?

---

## 🔍 왜 이 개념이 ML에서 중요한가

분할 기준은 결정트리의 **DNA**이다. (a) IG는 Information Theory와 ML을 잇는 가장 자연스러운 다리 — "feature가 label을 얼마나 결정하는가"의 정보량 측정, (b) ID3·C4.5·CART의 출발 알고리즘이고, (c) Random Forest·Gradient Boosting의 모든 트리가 같은 분할 메커니즘을 사용. 본 문서는 **"엔트로피를 ML에서 어떻게 쓰는가"의 첫 사례**이며, 이후 Gini (Ch3-02)·MSE 분할 (Ch3-03)·Boosting의 IG-like criteria로 확장된다. "feature 선택의 수학적 원리"를 정확히 이해하면 모든 트리 기반 모델이 명확해진다.

---

## 📐 수학적 선행 조건

- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): 엔트로피, 조건부 엔트로피, 상호정보량
- 확률의 곱법칙·조건부 확률
- 로그함수의 볼록성 (Jensen's inequality)

---

## 📖 직관적 이해

### 엔트로피 = "평균 불확실성"

$Y \in \{1, \ldots, K\}$, $p_k = P(Y = k)$. 엔트로피

$$H(Y) := -\sum_k p_k \log p_k.$$

- 모든 $p_k$가 같음 (uniform): $H = \log K$ (최대 — 가장 불확실).
- 한 $p_k = 1$, 나머지 0 (deterministic): $H = 0$ (최소 — 완전 확실).

직관: $Y$의 outcome을 message로 encode할 때 평균적으로 필요한 bit 수.

### Information Gain = 엔트로피의 평균 감소

데이터 $S$를 feature $A$ 값별로 분할 → $\{S_v\}_v$. 분할 후 평균 엔트로피:

$$H(S | A) = \sum_v \frac{|S_v|}{|S|} H(S_v).$$

Information Gain:

$$IG(S, A) := H(S) - H(S | A).$$

"$A$를 알면 $Y$의 불확실성이 얼마나 줄어드는가". 큰 IG → $A$가 $Y$를 잘 설명.

### IG = 상호정보량

$I(Y; A) := H(Y) - H(Y | A)$는 정의 그대로 IG. 즉 **"feature와 label 사이의 정보량"** 측정 — Information Theory의 표준 도구가 ML에 그대로 흘러왔다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 엔트로피

이산 확률변수 $Y$의 분포 $p_k = P(Y = k)$에 대해

$$H(Y) := -\sum_{k=1}^K p_k \log p_k \quad (\text{단위: nats with } \log = \ln, \text{ bits with } \log_2).$$

(컨벤션: $0 \log 0 := 0$.)

### 정의 1.2 — 조건부 엔트로피

$$H(Y | A) := \sum_{a} P(A = a) H(Y | A = a) = \sum_a P(A = a) \bigl[-\sum_k P(Y = k | A = a) \log P(Y = k | A = a)\bigr].$$

### 정의 1.3 — 상호정보량

$$I(Y; A) := H(Y) - H(Y | A) = \sum_{a, k} P(A = a, Y = k) \log \frac{P(A = a, Y = k)}{P(A = a) P(Y = k)}.$$

### 정의 1.4 — Sample Information Gain

데이터 $S = \{(x_i, y_i)\}$, feature $A$ (값 집합 $\mathcal{V}_A$). $S_v := \{i \in S : A(x_i) = v\}$. **Empirical entropy**:

$$\hat{H}(S) := -\sum_k \hat{p}_k \log \hat{p}_k, \quad \hat{p}_k = \frac{|\{i : y_i = k\}|}{|S|}.$$

**Information Gain**:

$$IG(S, A) := \hat{H}(S) - \sum_{v \in \mathcal{V}_A} \frac{|S_v|}{|S|} \hat{H}(S_v).$$

---

## 🔬 정리와 증명

### 정리 1.1 — 엔트로피는 비음 + 균등에서 최대

**명제**: $H(Y) \geq 0$, 등호는 deterministic ($p_k = 1$ for some $k$). $H(Y) \leq \log K$, 등호는 uniform ($p_k = 1/K$).

**증명**: 비음 — $-p_k \log p_k \geq 0$ for $p_k \in [0, 1]$.

상한 — Jensen's inequality: $H(Y) = \mathbb{E}[-\log p(Y)] \leq -\log \mathbb{E}[p(Y)]^{-1}$? 더 직접적으로: KL divergence $KL(p \| u) = \sum p_k \log(p_k / (1/K)) = \log K - H(p) \geq 0$ → $H(p) \leq \log K$. 등호는 $p = u$. $\square$

### 정리 1.2 — Information Gain = Mutual Information (모집단)

**명제**: 정의 1.4의 sample IG의 모집단판이 정확히 $I(Y; A)$.

**증명**: 정의 1.3을 펼치면

$$I(Y; A) = \sum_a P(A = a) [H(Y) - H(Y | A = a)] = H(Y) - \sum_a P(A = a) H(Y | A = a) = H(Y) - H(Y | A).$$

마지막은 정의 그대로 IG의 모집단 버전. $\square$

> 💡 **함의**: ID3의 IG 기준은 **"feature와 label 사이의 mutual information을 최대화"** — Information Theory의 정확한 ML 적용.

### 정리 1.3 — IG는 항상 ≥ 0

**명제**: $IG(S, A) \geq 0$. 등호는 $A$와 $Y$가 독립일 때만.

**증명**: $I(Y; A) \geq 0$ (KL divergence의 non-negativity); 등호는 $P(A, Y) = P(A) P(Y)$ — 독립. $\square$

> 📌 **결과**: 분할은 **항상 entropy를 감소**시킨다 (또는 변화 없음). "분할이 도움 안 됨" = "그 feature가 label과 독립".

### 정리 1.4 — IG의 High-Cardinality Bias

**명제**: $A$가 매우 많은 값을 가지면 (극단적으로 unique ID) $IG(S, A)$가 부풀려진다. 극단적으로 ID feature는 모든 sample을 자기 buckets로 분리 → $H(S_v) = 0$ → $IG = H(S)$ 최대.

**증명 스케치**: 각 $v$에 대해 $|S_v| = 1$ ($A$가 ID처럼 행동)이면 $H(S_v) = 0$ → $\sum \frac{|S_v|}{|S|} H(S_v) = 0$ → $IG = H(S)$. 그러나 이는 **overfitting** — test에 일반화 안 됨.

→ ID3의 약점. C4.5의 **Gain Ratio**가 해결책:

$$GR(S, A) := \frac{IG(S, A)}{H(A)},$$

여기서 $H(A) = -\sum_v \frac{|S_v|}{|S|} \log \frac{|S_v|}{|S|}$는 분할 자체의 entropy. ID 같은 high-cardinality feature는 $H(A)$가 매우 커서 GR이 작아짐 → 페널티. $\square$

### 정리 1.5 — ID3 알고리즘 (탐욕 분할)

**알고리즘**:

```
function ID3(S, features):
    if S is pure (모든 y_i 같음): return Leaf(majority class)
    if features 비어 있음: return Leaf(majority class)
    
    A* = argmax_{A in features} IG(S, A)
    Tree = Node(A*)
    for each value v of A*:
        S_v = {(x, y) in S : A*(x) = v}
        if S_v 비어 있음: Tree.child[v] = Leaf(majority of S)
        else: Tree.child[v] = ID3(S_v, features \ {A*})
    return Tree
```

**복잡도**: 깊이 $d$, feature 수 $p$, 데이터 $n$ → $O(n p d)$ for unique categorical features. CART (continuous features)는 $O(n p d \log n)$ — 각 split point 정렬.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(42)

def entropy(y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p + 1e-12))

def information_gain(X_col, y):
    H_S = entropy(y)
    H_cond = 0
    for v in np.unique(X_col):
        mask = (X_col == v)
        H_cond += np.sum(mask) / len(y) * entropy(y[mask])
    return H_S - H_cond

# ─────────────────────────────────────────────
# 1. 단순 예시 — Outlook/Play tennis (Mitchell ML 교재 고전)
# ─────────────────────────────────────────────
data = [
    ('Sunny',    'Hot',  'High',   'Weak',   'No'),
    ('Sunny',    'Hot',  'High',   'Strong', 'No'),
    ('Overcast', 'Hot',  'High',   'Weak',   'Yes'),
    ('Rain',     'Mild', 'High',   'Weak',   'Yes'),
    ('Rain',     'Cool', 'Normal', 'Weak',   'Yes'),
    ('Rain',     'Cool', 'Normal', 'Strong', 'No'),
    ('Overcast', 'Cool', 'Normal', 'Strong', 'Yes'),
    ('Sunny',    'Mild', 'High',   'Weak',   'No'),
    ('Sunny',    'Cool', 'Normal', 'Weak',   'Yes'),
    ('Rain',     'Mild', 'Normal', 'Weak',   'Yes'),
    ('Sunny',    'Mild', 'Normal', 'Strong', 'Yes'),
    ('Overcast', 'Mild', 'High',   'Strong', 'Yes'),
    ('Overcast', 'Hot',  'Normal', 'Weak',   'Yes'),
    ('Rain',     'Mild', 'High',   'Strong', 'No'),
]
features = ['Outlook', 'Temp', 'Humidity', 'Wind']
arr = np.array(data)
y = arr[:, -1]

print(f'전체 H(S) = {entropy(y):.4f}  bits')
print(f'(p_yes = {(y == "Yes").mean():.3f}, p_no = {(y == "No").mean():.3f})\n')
for j, name in enumerate(features):
    ig = information_gain(arr[:, j], y)
    print(f'IG({name}) = {ig:.4f}')

# ─────────────────────────────────────────────
# 2. Gain Ratio (high-cardinality bias 보정)
# ─────────────────────────────────────────────
def gain_ratio(X_col, y):
    ig = information_gain(X_col, y)
    H_split = entropy(X_col)   # 분할 자체의 entropy
    return ig / (H_split + 1e-12)

print(f'\nGain Ratios:')
for j, name in enumerate(features):
    gr = gain_ratio(arr[:, j], y)
    print(f'GR({name}) = {gr:.4f}')

# ─────────────────────────────────────────────
# 3. High-cardinality bias 시연 (정리 1.4)
# ─────────────────────────────────────────────
# ID-like feature 추가
np.random.seed(0)
n = 100
X_real = rng.integers(0, 3, n)            # 3개 카테고리
X_id   = np.arange(n)                      # n개 unique values
y_rand = (rng.uniform(size=n) < 0.6).astype(int)

ig_real = information_gain(X_real.astype(str), y_rand)
ig_id   = information_gain(X_id.astype(str), y_rand)
gr_real = gain_ratio(X_real.astype(str), y_rand)
gr_id   = gain_ratio(X_id.astype(str), y_rand)

print(f'\n3-카테고리 feature : IG = {ig_real:.3f}, GR = {gr_real:.3f}')
print(f'ID-like feature    : IG = {ig_id:.3f},  GR = {gr_id:.3f}')
print(f'(IG는 ID feature를 선호 — high-cardinality bias)')
print(f'(GR은 H(A) 페널티로 정상화)')

# ─────────────────────────────────────────────
# 4. sklearn 비교 — 같은 데이터에서 분할 first feature 확인
# ─────────────────────────────────────────────
# sklearn은 numeric만 받음 → label encode
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X_enc = enc.fit_transform(arr[:, :-1])
y_enc = (arr[:, -1] == 'Yes').astype(int)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0)
clf.fit(X_enc, y_enc)
print(f'\nsklearn (entropy, max_depth=1) 첫 분할 feature: {features[clf.tree_.feature[0]]}')
print(f'우리 IG 계산에서 max IG feature        : '
      f'{features[np.argmax([information_gain(arr[:, j], y) for j in range(4)])]}')
```

**출력 예시**:
```
전체 H(S) = 0.9403  bits
(p_yes = 0.643, p_no = 0.357)

IG(Outlook) = 0.2467
IG(Temp) = 0.0292
IG(Humidity) = 0.1518
IG(Wind) = 0.0481

Gain Ratios:
GR(Outlook) = 0.1564
GR(Temp) = 0.0188
GR(Humidity) = 0.1518
GR(Wind) = 0.0488

3-카테고리 feature : IG = 0.020, GR = 0.013
ID-like feature    : IG = 0.971, GR = 0.146
(IG는 ID feature를 선호 — high-cardinality bias)
(GR은 H(A) 페널티로 정상화)

sklearn (entropy, max_depth=1) 첫 분할 feature: Outlook
우리 IG 계산에서 max IG feature        : Outlook
```

---

## 🔗 실전 활용

- **ID3 / C4.5**: Quinlan 1986/1993 — 처음 잘 알려진 트리 알고리즘.
- **CART** (Breiman 1984): 같은 IG/Gini idea를 binary split + continuous feature로 확장. sklearn 기본.
- **sklearn `DecisionTreeClassifier(criterion='entropy')`**: ID3의 IG와 같은 기준.
- **Random Forest의 분할**: 같은 IG (또는 Gini) — 트리 마다 random feature subset만 후보 (Ch4-03).
- **Information Gain Ratio**: high-cardinality feature가 많은 데이터 (예: customer ID)에서 권장.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Categorical 가정 | 연속 feature는 binary threshold로 변환 (CART) |
| 탐욕 (greedy) | 최적 트리 NP-hard, 탐욕은 sub-optimal 가능 |
| High-cardinality bias | 정리 1.4 — Gain Ratio 또는 Gini로 완화 |
| Overfitting | 어느 feature든 충분 데이터에서 IG > 0 — 가지치기 필요 (Ch3-04) |

---

## 📌 핵심 정리

$$\boxed{IG(S, A) = H(S) - \sum_v \frac{|S_v|}{|S|}H(S_v) = I(Y; A) \geq 0}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **Entropy** | $H = -\sum p_k \log p_k$, 균등 ↔ 최대, deterministic ↔ 0 |
| **Conditional Entropy** | $H(Y\|A) = \sum P(A=a) H(Y\|A=a)$ |
| **IG = MI** | feature와 label의 상호정보량 |
| **High-card bias** | unique-value feature가 부당히 우세 — Gain Ratio 보정 |
| **ID3** | greedy IG 최대화로 트리 자라기 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $K = 2$일 때 $H(p, 1-p)$의 그래프와 최대점을 구하라.

<details>
<summary>힌트 및 해설</summary>

$H(p) = -p \log p - (1-p) \log(1-p)$. $p \in [0, 1]$, $H(0) = H(1) = 0$. 미분 = 0: $-\log p + \log(1-p) = 0$ → $p = 1/2$, $H_{\max} = \log 2 = 1$ bit. **공정한 동전이 가장 불확실**.

대칭 ($H(p) = H(1-p)$). 이것이 binary classification에서 entropy 그래프가 종 모양인 이유.

</details>

**문제 2** (심화): $A$가 binary feature ($A \in \{0, 1\}$)이고 split이 $|S_0| = |S_1| = n/2$이면 $IG(S, A)$의 최대값은? 어떤 조건에서 그 최대 달성?

<details>
<summary>힌트 및 해설</summary>

$H(S) \leq 1$ (bit). $IG = H(S) - \sum \frac{1}{2} H(S_v) = H(S) - \frac{1}{2}(H(S_0) + H(S_1))$. 최대 IG = $H(S) = 1$ when $H(S_0) = H(S_1) = 0$ — 두 split 모두 pure (한 클래스만).

→ **완벽 분리**: 이것이 ID3의 첫 split이 가장 강하게 분리되는 binary feature를 선택하는 이유.

</details>

**문제 3** (ML 연결): NN의 cross-entropy loss $-\sum y_k \log p_k$는 sample IG와 어떻게 다른가? 둘이 같은 정보 이론적 객체를 다른 방식으로 다루는 점을 설명하라.

<details>
<summary>힌트 및 해설</summary>

**Sample IG**: $H(Y) - H(Y|A)$. Population entropy의 추정 — feature가 label을 얼마나 결정하는가.

**NN의 CE loss**: $-\sum y_k \log p_k$ — predicted distribution $p$가 true (one-hot) $y$를 얼마나 잘 예측하나. 이는 **KL($y \| p$)** = $H(y, p)$ - $H(y) = $ CE (since one-hot의 entropy = 0).

두 도구 모두 **entropy 기반**:
- IG: feature 선택 (이산)
- CE: continuous parameter 학습

NN의 CE를 minimize = predicted $p$의 entropy를 데이터에 맞게 줄이는 것 = "데이터가 model에 더 정확한 정보를 주게 하는" 것. **두 객체 모두 information theory의 도구**, 다른 단위 (sample IG는 bit, NN의 nats).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch2-05. Separation·Firth](../ch2-logistic-glm/05-separation-firth.md) | [📚 README](../README.md) | [02. Gini Impurity vs Entropy ▶](./02-gini-vs-entropy.md) |

</div>
