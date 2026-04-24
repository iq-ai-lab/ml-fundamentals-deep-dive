# 04. Multinomial / Softmax Regression

## 🎯 핵심 질문

- Multinomial Regression의 두 표현 — **One-vs-Rest**(K개의 binary LR)와 **직접 multinomial** (softmax)는 어떻게 다른가?
- Softmax $p_k = e^{w_k^\top x}/\sum_j e^{w_j^\top x}$의 **identifiability 문제**(weight를 모두 더해도 같은 확률)는 어떻게 해결하는가? Reference class 고정의 의미는?
- 왜 **Cross-Entropy Loss**가 정확히 **Categorical MLE의 negative log-likelihood**인가?
- Newton-Raphson을 직접 구현해서 sklearn `LogisticRegression(multi_class='multinomial')`과 일치 검증.

---

## 🔍 왜 이 개념이 ML에서 중요한가

Softmax + Cross-Entropy는 **모든 multi-class 분류기의 표준 출력층** — sklearn의 multinomial LR, 모든 NN의 분류 head(ResNet·BERT·GPT 분류기), 강화학습의 policy 분포까지 모두 같은 식. 본 문서는 (a) **categorical MLE → softmax**의 자연스러운 유도, (b) softmax의 identifiability와 그것이 NN의 학습에 미치는 영향, (c) **cross-entropy = MLE = KL divergence**의 세 가지 동치, 그리고 (d) **multinomial Newton-Raphson의 IRLS-like 구현**까지 한 번에 정리한다. "왜 마지막 layer가 softmax + CE인가"를 모르고 NN을 짜는 것은 차의 핸들이 왜 둥근지 모르고 운전하는 것과 같다.

---

## 📐 수학적 선행 조건

- LR의 MLE와 IRLS (Ch2-01, Ch2-02)
- GLM과 canonical link (Ch2-03)
- Categorical 분포, KL divergence

---

## 📖 직관적 이해

### Softmax = Logit의 K-class 일반화

이진: $\log \frac{P(y=1)}{P(y=0)} = w^\top x$.

K-class: $\log \frac{P(y=k)}{P(y=K)} = w_k^\top x$ ($k = 1, \ldots, K-1$). $w_K = 0$ (reference). 

$P(y=k) = \frac{e^{w_k^\top x}}{\sum_{j=1}^K e^{w_j^\top x}}$ — **softmax**. K=2이면 sigmoid 회복.

### Cross-Entropy의 의미

True 분포 $q(y) = \mathbb{1}[y = y_{\text{true}}]$ (one-hot). Predicted $p(y) = \text{softmax}_k$. 두 분포의 cross-entropy:

$$H(q, p) = -\sum_k q_k \log p_k = -\log p_{y_{\text{true}}}.$$

이를 데이터 전체에 합하면 **negative log-likelihood**. $\text{KL}(q \| p) = H(q, p) - H(q) = H(q, p)$ (one-hot의 entropy 0). 따라서 **CE 최소화 = KL 최소화 = MLE**.

### Identifiability 문제

$w_k \to w_k + c$ (모든 클래스에 같은 $c$ 더함)이면 $p_k$ 불변 (분자·분모에서 $e^{c^\top x}$ 약분). → **해의 비유일성**. **해결**: $w_K = 0$ 고정 (reference class) 또는 sum-to-zero 제약 또는 L2 regularization.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Softmax Function

$z = (z_1, \ldots, z_K) \in \mathbb{R}^K$에 대해 

$$\text{softmax}(z)_k := \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad k = 1, \ldots, K.$$

성질: $\text{softmax}(z)_k > 0$, $\sum_k \text{softmax}(z)_k = 1$ — 확률 분포.

### 정의 4.2 — Multinomial Logistic Regression

$y \in \{1, \ldots, K\}$. 모델

$$P(y = k \mid x; W) = \text{softmax}(W x)_k = \frac{e^{w_k^\top x}}{\sum_{j=1}^K e^{w_j^\top x}}, \quad W = [w_1, \ldots, w_K]^\top \in \mathbb{R}^{K \times p}.$$

### 정의 4.3 — Categorical Cross-Entropy Loss

데이터 $\{(x_i, y_i)\}$, one-hot encoding $y_{ik} = \mathbb{1}[y_i = k]$. Loss:

$$\mathcal{L}(W) = -\sum_{i=1}^n \sum_{k=1}^K y_{ik} \log p_{ik} = -\sum_i \log p_{i, y_i}.$$

---

## 🔬 정리와 증명

### 정리 4.1 — Cross-Entropy = Multinomial MLE

**명제**: 정의 4.2의 모델 하에서 $-\log L(W) = \mathcal{L}(W)$.

**증명**: $L(W) = \prod_i p_{i, y_i} = \prod_i \prod_k p_{ik}^{y_{ik}}$ (one-hot encoding). $\log L = \sum_i \sum_k y_{ik} \log p_{ik}$. 음수 → CE. $\square$

> 💡 **연결**: KL($q \| p$) where $q$ one-hot. $H(q) = 0$이므로 KL = CE. 즉 **CE 최소화 = KL 최소화 = MLE**.

### 정리 4.2 — Softmax의 Gradient

**명제**: $\nabla_{w_k} \mathcal{L} = \sum_i (p_{ik} - y_{ik}) x_i = X^\top (p_k - y_k)$ where $p_k$는 모든 점의 클래스 k 확률.

**증명**: $\partial \mathcal{L} / \partial z_{ik}$ where $z_{ik} = w_k^\top x_i$. 

$\partial p_{ij} / \partial z_{ik} = p_{ij}(\delta_{jk} - p_{ik})$ (softmax 도함수, 직접 계산).

$\partial \mathcal{L} / \partial z_{ik} = -\sum_j y_{ij} \cdot (\delta_{jk} - p_{ik}) = -y_{ik} + \sum_j y_{ij} p_{ik} = -y_{ik} + p_{ik}$ ($\sum_j y_{ij} = 1$).

$\partial z_{ik} / \partial w_k = x_i$. → $\nabla_{w_k} \mathcal{L} = \sum_i (p_{ik} - y_{ik}) x_i$. $\square$

> 📌 **OLS·LR 일관성**: gradient = "예측 - 정답"의 데이터로의 사영. 모든 GLM의 공통 패턴.

### 정리 4.3 — Identifiability와 Reference Class

**명제**: $W \mapsto W + \mathbf{1} c^\top$ ($c \in \mathbb{R}^p$, $\mathbf{1}$은 K차원)로 다른 $W'$를 만들면 $\text{softmax}(W'x) = \text{softmax}(Wx)$.

**증명**: $w'_k = w_k + c$ → $z'_k = z_k + c^\top x$. 분자 $e^{z'_k} = e^{z_k} e^{c^\top x}$, 분모 $\sum_j e^{z'_j} = e^{c^\top x} \sum_j e^{z_j}$ → $p'_k = p_k$. $\square$

**해결**: $w_K := 0$ 고정 → identifiable. 이때 모델은 사실상 $K-1$개 weight vector를 학습. sklearn `LogisticRegression(multi_class='multinomial')`은 모든 K개 weight를 학습하지만 결과는 sum-to-zero 또는 L2 regularization으로 unique하게 fix.

### 정리 4.4 — Cross-Entropy는 Convex (in $W$)

**명제**: $\mathcal{L}(W)$는 $W$의 affine 함수에 대한 log-sum-exp의 합 — 볼록.

**증명**: $\mathcal{L} = \sum_i [-y_i^\top z_i + \log \sum_k e^{z_{ik}}]$ where $z_i = W x_i$.

- 첫 항 $-y_i^\top z_i$는 $W$에 대한 linear → convex (실제로 affine).
- 두 번째 항 $\log\sum_k e^{z_{ik}}$는 **log-sum-exp** — 볼록 (잘 알려진 결과).
- 합 → 볼록. $\square$

> 💡 **결과**: GD/Newton 모두 unique global min에 수렴. NN의 마지막 layer만 보면 convex (개별 layer는 볼록), 전체 NN은 비볼록.

### 정리 4.5 — Newton-Raphson Block Hessian

**명제**: Hessian은 $K p \times K p$ 블록 행렬 — $(j, k)$ 블록 = $X^\top D_{jk} X$, where $D_{jk} = \text{diag}(p_{ij}(\delta_{jk} - p_{ik}))$. 

전체 Hessian의 음반정치성 (정리 4.4의 또 다른 증명).

**증명 스케치**: $\partial^2 \mathcal{L} / \partial z_{ij} \partial z_{ik} = p_{ij}(\delta_{jk} - p_{ik})$ → 위 블록 형태. PSD는 softmax의 covariance matrix structure. 자세히는 Bishop PRML 4.3.4. $\square$

### 정리 4.6 — One-vs-Rest vs Multinomial 비교

**One-vs-Rest (OvR)**: K개 binary LR을 별도 학습. 각 $w_k$가 "$y = k$ vs $y \neq k$"를 분류.

**Multinomial**: 한 번의 joint MLE. $W$ 전체를 함께 최적화.

**차이**:
- OvR은 $\sum_k p_k \neq 1$ — 후처리로 normalize 필요.
- Multinomial은 $\sum_k p_k = 1$ 자동 보장.
- 이론적으로 multinomial이 더 calibrated.
- 실무에서 보통 multinomial이 더 정확하나 큰 K에서는 OvR이 빠름.

sklearn `multi_class='ovr'` (구버전 default) vs `'multinomial'` (v1.0+ default).

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

rng = np.random.default_rng(42)

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)   # 수치 안정 (overflow 방지)
    eZ = np.exp(Z)
    return eZ / eZ.sum(axis=1, keepdims=True)

# ─────────────────────────────────────────────
# 1. Multinomial NR 구현
# ─────────────────────────────────────────────
def multinomial_nr(X, Y, K, n_iter=100, tol=1e-8, ridge=0.01):
    """
    Y: one-hot (n, K)
    Returns W: (K, p)
    """
    n, p = X.shape
    W = np.zeros((K, p))
    for it in range(n_iter):
        Z = X @ W.T                          # (n, K)
        P = softmax(Z)                        # (n, K)
        # Gradient (Kp,)
        grad = ((P - Y).T @ X).flatten()      # (K, p) → (Kp,)
        # Hessian: Kronecker structure (Kp, Kp)
        H = np.zeros((K * p, K * p))
        for j in range(K):
            for k in range(K):
                d = P[:, j] * ((j == k) - P[:, k])
                H[j*p:(j+1)*p, k*p:(k+1)*p] = X.T @ (d[:, None] * X)
        H += ridge * np.eye(K * p)
        try:
            update = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        W_new = W - update.reshape(K, p)
        if np.linalg.norm(W_new - W) < tol:
            break
        W = W_new
    return W, it + 1

# ─────────────────────────────────────────────
# 2. Iris dataset
# ─────────────────────────────────────────────
data = load_iris()
X = (data.data - data.data.mean(0)) / data.data.std(0)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # bias
y = data.target
K = 3
Y_one_hot = np.eye(K)[y]

W_nr, iters = multinomial_nr(X, Y_one_hot, K, ridge=0.01)
print(f'Multinomial NR W (수렴 {iters} iter):')
print(W_nr.round(3))

# Predict
P_nr = softmax(X @ W_nr.T)
y_pred_nr = P_nr.argmax(axis=1)
print(f'\n정확도 (NR): {(y_pred_nr == y).mean():.4f}')

# sklearn 비교
sk = LogisticRegression(multi_class='multinomial', solver='newton-cg',
                        C=100, max_iter=2000, fit_intercept=False).fit(X, y)
P_sk = sk.predict_proba(X)
print(f'정확도 (sklearn): {sk.score(X, y):.4f}')

# probabilities 비교 (identifiability 때문에 W는 다를 수 있음)
print(f'\nP_nr vs P_sk 평균 차이: {np.mean(np.abs(P_nr - P_sk)):.4f}')

# ─────────────────────────────────────────────
# 3. Identifiability 시연 (정리 4.3)
# ─────────────────────────────────────────────
c = rng.standard_normal(X.shape[1])
W_shifted = W_nr + np.outer(np.ones(K), c)
P_shifted = softmax(X @ W_shifted.T)
print(f'\n||P_nr - P_shifted|| = {np.linalg.norm(P_nr - P_shifted):.2e}')
print(f'(W에 c를 더해도 P 불변 — identifiability 문제)')

# ─────────────────────────────────────────────
# 4. Cross-Entropy = MLE 검증 (정리 4.1)
# ─────────────────────────────────────────────
def cross_entropy(Y, P):
    return -np.sum(Y * np.log(P + 1e-12)) / len(Y)

def neg_log_likelihood(Y, P):
    return -np.sum(np.log(P[np.arange(len(Y)), Y.argmax(axis=1)] + 1e-12)) / len(Y)

ce = cross_entropy(Y_one_hot, P_nr)
nll = neg_log_likelihood(Y_one_hot, P_nr)
print(f'\nCross-Entropy : {ce:.4f}')
print(f'Neg log-L     : {nll:.4f}')
print(f'(같은 값임을 확인 — 정리 4.1)')
```

**출력 예시**:
```
Multinomial NR W (수렴 7 iter):
[[ 0.13   0.32  -0.18   0.34   0.19]
 [ 0.07  -0.43   1.32  -0.85  -0.95]
 [-0.20   0.11  -1.14   0.51   0.76]]

정확도 (NR): 0.9800
정확도 (sklearn): 0.9800

P_nr vs P_sk 평균 차이: 0.0021

||P_nr - P_shifted|| = 1.36e-15
(W에 c를 더해도 P 불변 — identifiability 문제)

Cross-Entropy : 0.0814
Neg log-L     : 0.0814
(같은 값임을 확인 — 정리 4.1)
```

---

## 🔗 실전 활용

- **모든 NN 분류기의 출력층**: PyTorch `CrossEntropyLoss` = LogSoftmax + NLLLoss = multinomial LR의 NLL.
- **Multi-label 분류**: 여러 정답 가능 (예: 영화 장르). softmax 대신 **K개 sigmoid** + binary CE.
- **Hierarchical softmax**: K가 매우 큰 경우 (NLP의 vocab 30,000) 계산 빠르게 — 트리 구조로 분해.
- **Sampled softmax / NCE**: K가 매우 큰 경우 일부만 sampling, word2vec 학습.
- **Temperature scaling**: $\text{softmax}(z/T)$, $T > 1$이면 분포 부드러워짐 — calibration·distillation.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Class prior 균형 | imbalanced면 majority class에 편향 — class_weight 보정 |
| Independent 분류 | classes 간 hierarchy 무시 — hierarchical softmax 또는 ordinal regression |
| Calibration | overconfident 출력 흔함 — temperature scaling, Platt |
| 큰 K | $O(K)$ 비용 — sampled softmax |
| Identifiability | 무 정규화면 unique 아님 — L2 또는 ref class 필요 |

---

## 📌 핵심 정리

$$\boxed{P(y = k \mid x) = \text{softmax}(Wx)_k,\ \mathcal{L}_{\text{CE}} = -\sum y_k \log p_k = -\log L = \text{KL}(q \| p)}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **Softmax** | sigmoid의 K-class 일반화, $\sum p_k = 1$ |
| **Cross-Entropy** | Categorical MLE의 negative log-likelihood |
| **Identifiability** | $W \to W + \mathbf{1}c^\top$로 $P$ 불변 — ref class 또는 L2로 해결 |
| **Convexity** | $\mathcal{L}(W)$는 $W$에 대해 convex (log-sum-exp의 합) |
| **Gradient** | $X^\top (P_k - Y_k)$ — 모든 GLM의 공통 패턴 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): K = 2일 때 softmax가 sigmoid로 환원됨을 보여라.

<details>
<summary>힌트 및 해설</summary>

K = 2: $p_1 = e^{z_1}/(e^{z_1} + e^{z_2})$, $p_2 = e^{z_2}/(e^{z_1} + e^{z_2})$.

$p_1 = 1/(1 + e^{z_2 - z_1}) = \sigma(z_1 - z_2)$. 

Identifiability ($w_2 := 0$로 고정): $z_2 = 0$ → $p_1 = \sigma(z_1) = \sigma(w_1^\top x)$ — binary LR.

</details>

**문제 2** (심화): Cross-Entropy의 gradient $\partial \mathcal{L} / \partial z_k = p_k - y_k$를 직접 유도하라 (chain rule 사용).

<details>
<summary>힌트 및 해설</summary>

$\mathcal{L} = -\sum_j y_j \log p_j$. $\partial \log p_j / \partial z_k = (1/p_j) \partial p_j / \partial z_k = (1/p_j) \cdot p_j (\delta_{jk} - p_k) = \delta_{jk} - p_k$.

$\partial \mathcal{L} / \partial z_k = -\sum_j y_j (\delta_{jk} - p_k) = -y_k + p_k \sum_j y_j = -y_k + p_k$ ($\sum y_j = 1$).

→ $\partial \mathcal{L} / \partial z_k = p_k - y_k$. **매우 깔끔**: gradient가 단순히 "예측 - 정답". 이것이 NN backprop의 마지막 layer가 매우 단순한 이유.

</details>

**문제 3** (ML 연결): Distillation에서 student가 teacher의 soft target을 모방할 때 cross-entropy를 쓰는 이유를 KL divergence 관점에서 설명하라.

<details>
<summary>힌트 및 해설</summary>

Teacher의 출력 $q$ (full distribution, not one-hot), student의 출력 $p$. Distillation loss = CE($q$, $p$) = KL($q \| p$) + H($q$). H($q$)는 student-independent → CE 최소화 = KL($q \| p$) 최소화.

**핵심**: One-hot label로 학습하면 H(q) = 0이라 KL = CE. Soft label로 학습하면 H(q) > 0 — student는 추가 정보 (다른 class들의 상대적 유사성)도 학습. 이것이 Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"의 핵심 아이디어.

**Temperature**: $\text{softmax}(z/T)$로 분포를 부드럽게 만들면 더 풍부한 정보 전달. $T \to \infty$이면 uniform, $T = 1$이면 표준 softmax.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Exponential Family](./03-exp-family-canonical-link.md) | [📚 README](../README.md) | [05. 분리 문제와 Firth Correction ▶](./05-separation-firth.md) |

</div>
