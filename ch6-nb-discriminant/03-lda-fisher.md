# 03. LDA의 Fisher Discriminant 해석

## 🎯 핵심 질문

- Fisher의 LDA: $J(w) = \frac{w^\top S_B w}{w^\top S_W w}$ — between-class / within-class variance 비를 최대화.
- 이 최적화는 **일반화 고유값 문제** $S_B w = \lambda S_W w$의 해와 동치임을 어떻게 증명하는가?
- LDA를 **차원축소** 도구로 사용: $K - 1$ 차원으로 projection.
- **PCA와의 차이**: PCA는 분산 최대 (unsupervised), LDA는 클래스 분리 최대 (supervised).

---

## 🔍 왜 이 개념이 ML에서 중요한가

Fisher LDA (1936)는 (a) **supervised dimensionality reduction의 표준** — PCA의 분류 친화 버전, (b) **클래스 분리도 최대화**의 직관적 정식화 — visualization·preprocessing, (c) **PCA·CCA·t-SNE·UMAP과 같은 spectral 차원축소의 family**, (d) **Fisher 추정량의 통계학적 우아함** — 일반화 고유값 문제. 본 문서는 Ch6-02에서 본 LDA를 **classification 관점**이 아닌 **차원축소 관점**으로 재해석.

---

## 📐 수학적 선행 조건

- LDA의 정의 (Ch6-02)
- 행렬 미적분, Lagrange multiplier
- 일반화 고유값 문제 $A v = \lambda B v$
- PCA의 SVD 유도

---

## 📖 직관적 이해

### 좋은 projection이란?

데이터를 1차원 line $w^\top x$로 projection. 좋은 line은:
- 같은 클래스 점들이 **뭉쳐 있음** (작은 within-class variance)
- 다른 클래스 점들이 **떨어져 있음** (큰 between-class variance)

→ 두 변량의 비 $J(w) = \frac{\sigma_{\text{between}}^2}{\sigma_{\text{within}}^2}$를 최대화.

### Scatter Matrix

**Within-class scatter** $S_W$: 각 클래스 내 분산의 합.

**Between-class scatter** $S_B$: 클래스 평균들 사이의 분산.

$$J(w) = \frac{w^\top S_B w}{w^\top S_W w}.$$

### 일반화 고유값 문제

$\nabla J = 0$ → $S_B w = \lambda S_W w$. 이는 일반화 eigenvalue. 가장 큰 $\lambda$의 $w$가 최적 projection 방향.

### PCA와의 차이

PCA: $\arg\max \frac{w^\top S w}{w^\top w}$ where $S$는 전체 covariance — **레이블 무시**.

LDA: $\arg\max \frac{w^\top S_B w}{w^\top S_W w}$ — 레이블 활용해 클래스 분리 최대.

→ **같은 데이터에서 PCA와 LDA가 완전히 다른 방향 선택**할 수 있음.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Class Mean and Overall Mean

클래스 $k$의 평균 $\mu_k = \frac{1}{n_k}\sum_{i: y_i = k} x_i$. 전체 평균 $\bar{\mu} = \frac{1}{n}\sum_i x_i$.

### 정의 3.2 — Within-Class Scatter

$$S_W := \sum_k \sum_{i: y_i = k}(x_i - \mu_k)(x_i - \mu_k)^\top.$$

### 정의 3.3 — Between-Class Scatter

$$S_B := \sum_k n_k (\mu_k - \bar{\mu})(\mu_k - \bar{\mu})^\top.$$

### 정의 3.4 — Total Scatter

$$S_T := \sum_i (x_i - \bar{\mu})(x_i - \bar{\mu})^\top = S_W + S_B.$$

(이것이 분산 분해: 전체 변동 = within-class + between-class.)

### 정의 3.5 — Fisher Discriminant Ratio

$$J(w) := \frac{w^\top S_B w}{w^\top S_W w}.$$

---

## 🔬 정리와 증명

### 정리 3.1 — Total = Within + Between (분산 분해)

**명제**: $S_T = S_W + S_B$.

**증명**: $x_i - \bar{\mu} = (x_i - \mu_{y_i}) + (\mu_{y_i} - \bar{\mu})$. 외적 전개:

$S_T = \sum_i (x_i - \bar{\mu})(x_i - \bar{\mu})^\top = S_W + 2 \cdot \text{cross} + \sum_i (\mu_{y_i} - \bar{\mu})(\mu_{y_i} - \bar{\mu})^\top$.

Cross term = $\sum_k \sum_{i: y_i = k}(x_i - \mu_k)(\mu_k - \bar{\mu})^\top + \text{transp}$. 안쪽 합 = $(\sum_{i: y_i = k}(x_i - \mu_k))(\mu_k - \bar{\mu})^\top = 0$ (편차합 = 0).

마지막 항 = $\sum_k n_k(\mu_k - \bar{\mu})(\mu_k - \bar{\mu})^\top = S_B$. → $S_T = S_W + S_B$. $\square$

> 💡 **함의**: 분산을 두 부분으로 나눔 — 클래스 내 변동 + 클래스 간 변동. ANOVA의 핵심 항등식.

### 정리 3.2 — Fisher 최적화는 일반화 고유값 문제

**명제**: $w^* = \arg\max_w J(w)$는 일반화 고유값 문제

$$S_B w = \lambda S_W w$$

의 가장 큰 $\lambda$에 대응하는 고유벡터.

**증명**: $J(w) = \frac{w^\top S_B w}{w^\top S_W w}$. Scale invariant → constraint $w^\top S_W w = 1$로 normalize.

Lagrangian: $L = w^\top S_B w - \lambda(w^\top S_W w - 1)$. $\nabla_w L = 2 S_B w - 2 \lambda S_W w = 0 \Rightarrow S_B w = \lambda S_W w$. $\square$

**함의**: $S_W$ 가역이면 $S_W^{-1} S_B w = \lambda w$ — 표준 고유값 문제. 가장 큰 $\lambda$의 eigenvector가 최적 방향.

### 정리 3.3 — 2-Class LDA의 Closed-Form

**명제**: 두 클래스에서 $S_B = n_1 n_2 / n \cdot (\mu_1 - \mu_2)(\mu_1 - \mu_2)^\top$ — rank 1. 따라서 $S_W^{-1} S_B$의 nonzero eigenvalue는 1개. 그 eigenvector:

$$w^* \propto S_W^{-1}(\mu_1 - \mu_2).$$

**증명**: $S_B w = \lambda S_W w \Rightarrow$ $S_B = c (\mu_1 - \mu_2)(\mu_1 - \mu_2)^\top$이므로 $S_B w \propto \mu_1 - \mu_2$. 따라서 $S_W^{-1}(\mu_1 - \mu_2)$가 eigenvector (방향). $\square$

> 📌 **Ch6-02의 LDA $w$와 같음**: 정리 6.1 Section 2.1의 LDA 분류기 $w = \Sigma^{-1}(\mu_1 - \mu_2)$. $\Sigma \approx S_W/(n - K)$이므로 본질적으로 같은 방향.

### 정리 3.4 — Multi-Class LDA: $K - 1$ Dimensions

**명제**: $K$개 클래스에서 $S_B$의 rank $\leq K - 1$. 따라서 일반화 고유값 문제의 nonzero eigenvalue 개수 $\leq K - 1$.

**증명**: $S_B = \sum_k n_k v_k v_k^\top$ where $v_k = \mu_k - \bar{\mu}$. $\sum_k n_k v_k = \sum n_k \mu_k - n \bar{\mu} = n \bar{\mu} - n \bar{\mu} = 0$ → $v_k$들이 선형종속 → rank ≤ $K - 1$. $\square$

**결론**: LDA는 $K - 1$차원 subspace로 최적 projection. Iris 3 클래스 → 2D LDA projection이 정확한 visualization.

### 정리 3.5 — LDA vs PCA

**LDA**: 

$$\arg\max_W \frac{|W^\top S_B W|}{|W^\top S_W W|}.$$

→ 클래스 분리 최대.

**PCA**:

$$\arg\max_W \frac{|W^\top S_T W|}{|W^\top W|} \quad \text{or} \quad \arg\max_W |W^\top S_T W| \text{ s.t. } W^\top W = I.$$

→ 분산 최대 (레이블 무관).

**예**: 모든 클래스 평균이 같음 ($\mu_k = \bar{\mu}$, between scatter 작음) — PCA는 클래스 무관 분산 큰 방향, LDA는 클래스 정보 없으므로 fail.

**역**: 클래스 평균은 다른데 within variance가 큰 노이지 데이터 — PCA는 노이즈 따라가지만 LDA가 정확히 클래스 분리.

---

## 💻 NumPy로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Fisher LDA 바닥 구현
# ─────────────────────────────────────────────
def fisher_lda(X, y, n_components=None):
    classes = np.unique(y)
    K = len(classes)
    n, p = X.shape
    if n_components is None:
        n_components = min(p, K - 1)
    
    # Class means and overall mean
    mu_overall = X.mean(axis=0)
    means = []
    for k in classes:
        means.append(X[y == k].mean(axis=0))
    means = np.array(means)
    
    # S_W (within-class scatter)
    S_W = np.zeros((p, p))
    for k_idx, k in enumerate(classes):
        diff = X[y == k] - means[k_idx]
        S_W += diff.T @ diff
    
    # S_B (between-class scatter)
    S_B = np.zeros((p, p))
    for k_idx, k in enumerate(classes):
        n_k = (y == k).sum()
        diff = (means[k_idx] - mu_overall).reshape(-1, 1)
        S_B += n_k * diff @ diff.T
    
    # 일반화 고유값 문제 S_B w = λ S_W w → S_W^-1 S_B w = λ w
    S_W_inv = np.linalg.inv(S_W + 1e-6 * np.eye(p))
    eigvals, eigvecs = np.linalg.eig(S_W_inv @ S_B)
    
    # 큰 λ 순으로 정렬
    order = np.argsort(-eigvals.real)
    eigvecs = eigvecs[:, order].real
    eigvals = eigvals[order].real
    
    return eigvecs[:, :n_components], eigvals[:n_components]

# Iris로 실험
data = load_iris()
X, y = data.data, data.target

W_my, lambda_my = fisher_lda(X, y, n_components=2)
print(f'My LDA eigenvalues: {lambda_my}')
print(f'(K - 1 = 2개의 nonzero λ — 정리 3.4)')

# sklearn 비교
sk_lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
X_lda_my = X @ W_my
X_lda_sk = sk_lda.transform(X)

# 두 결과가 같은 방향 (부호·scale 무시)
def cos_sim(a, b):
    return abs((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f'\n방향 유사도 (My LDA vs sklearn):')
for k in range(2):
    sim = cos_sim(X_lda_my[:, k], X_lda_sk[:, k])
    print(f'  Component {k+1}: {sim:.4f}')

# ─────────────────────────────────────────────
# 2. LDA vs PCA — Iris 시각화 비교
# ─────────────────────────────────────────────
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

print(f'\n2D projection 비교 (cluster 분리도 정량):')
for name, X_proj in [('PCA', X_pca), ('LDA', X_lda_sk)]:
    # 클래스 평균 분리 vs 클래스 내 분산
    means_proj = np.array([X_proj[y == k].mean(axis=0) for k in range(3)])
    between = np.var(means_proj, axis=0).sum()
    within = sum(np.var(X_proj[y == k], axis=0).sum() for k in range(3)) / 3
    print(f'  {name}: between/within = {between/within:.4f}')

# ─────────────────────────────────────────────
# 3. 2-class LDA의 closed form (정리 3.3)
# ─────────────────────────────────────────────
y_binary = (y == 0).astype(int)   # versicolor·virginica vs setosa
mu1 = X[y_binary == 1].mean(axis=0)
mu0 = X[y_binary == 0].mean(axis=0)

S_W_bin = np.zeros((4, 4))
for k in [0, 1]:
    diff = X[y_binary == k] - X[y_binary == k].mean(axis=0)
    S_W_bin += diff.T @ diff

w_optimal = np.linalg.solve(S_W_bin + 1e-6 * np.eye(4), mu1 - mu0)
print(f'\n2-class Fisher direction:')
print(f'  w = {w_optimal.round(4)}')
print(f'  |w| = {np.linalg.norm(w_optimal):.4f}')

# ─────────────────────────────────────────────
# 4. LDA가 PCA보다 나은 케이스 — 클래스 분리 정보가 분산 작은 방향
# ─────────────────────────────────────────────
n = 100
# Class 0: 큰 분산이 x_1 방향, 작은 분산이 x_2 방향
X1 = rng.standard_normal((n, 2)) @ np.diag([3, 0.3])
X1[:, 1] -= 1.5  # 약간 위로

# Class 1: 같은 분산 패턴, 다른 위치
X2 = rng.standard_normal((n, 2)) @ np.diag([3, 0.3])
X2[:, 1] += 1.5  # 아래로

X_demo = np.vstack([X1, X2])
y_demo = np.array([0]*n + [1]*n)

print(f'\n2D 데모 — class 분리는 x_2 방향, 분산은 x_1 방향 우세:')
pca_demo = PCA(n_components=1).fit(X_demo).components_[0]
W_demo, _ = fisher_lda(X_demo, y_demo, n_components=1)
print(f'  PCA top component: {pca_demo.round(4)} (x_1 우세 — 분산 큰 방향)')
print(f'  LDA component    : {W_demo[:, 0].round(4)} (x_2 우세 — 분류에 유용)')

# 두 방향의 분류 정확도
from sklearn.linear_model import LogisticRegression
acc_pca = LogisticRegression().fit(X_demo @ pca_demo[:, None], y_demo).score(X_demo @ pca_demo[:, None], y_demo)
acc_lda = LogisticRegression().fit(X_demo @ W_demo, y_demo).score(X_demo @ W_demo, y_demo)
print(f'  1D projection으로 LR 분류:')
print(f'    PCA 방향 acc: {acc_pca:.4f}')
print(f'    LDA 방향 acc: {acc_lda:.4f}')
```

**출력 예시**:
```
My LDA eigenvalues: [32.18 0.29]
(K - 1 = 2개의 nonzero λ — 정리 3.4)

방향 유사도 (My LDA vs sklearn):
  Component 1: 0.9998
  Component 2: 0.9985

2D projection 비교 (cluster 분리도 정량):
  PCA: between/within = 5.4321
  LDA: between/within = 38.2143

2-class Fisher direction:
  w = [-0.4523  0.6234 -1.8231 -0.5421]
  |w| = 2.1034

2D 데모 — class 분리는 x_2 방향, 분산은 x_1 방향 우세:
  PCA top component: [0.9821 -0.1834] (x_1 우세 — 분산 큰 방향)
  LDA component    : [0.0782  0.9969] (x_2 우세 — 분류에 유용)
  1D projection으로 LR 분류:
    PCA 방향 acc: 0.5800
    LDA 방향 acc: 1.0000
```

---

## 🔗 실전 활용

- **sklearn `LinearDiscriminantAnalysis`**: 분류 + 차원축소 둘 다.
- **Visualization**: $K = 3$인 경우 2D LDA projection.
- **Preprocessing**: 고차원 데이터에서 LDA → SVM/LR. 종종 PCA보다 정확.
- **Face recognition**: Fisherface (Belhumeur 1997) — 얼굴 인식의 LDA 응용.
- **Multi-class LDA**: $K$ 클래스 → $K-1$ 차원 — 차원축소의 자연스러운 한계.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Gaussian + 공유 cov | 정규성·공분산 동일 가정 — 깨지면 부정확 |
| Linear projection | 비선형 구조 표현 불가 — Kernel LDA로 일반화 가능 |
| $K - 1$ 차원 | 클래스 적으면 차원 매우 낮음 |
| $S_W$ singular | high-dim ($p > n$)에서 — Regularized LDA 또는 PCA 후 LDA |

---

## 📌 핵심 정리

$$\boxed{w^* = \arg\max \frac{w^\top S_B w}{w^\top S_W w} \text{ from } S_B w = \lambda S_W w; \text{2-class: } w \propto S_W^{-1}(\mu_1 - \mu_2)}$$

| 개념 | 한 줄 요약 |
|------|-----------|
| **분산 분해** | $S_T = S_W + S_B$ (ANOVA 항등식) |
| **Fisher ratio** | between/within 분산 비 |
| **일반화 고유값** | $S_B w = \lambda S_W w$ 푸는 것 |
| **$K - 1$ 차원** | $S_B$의 rank가 $K - 1$ |
| **LDA vs PCA** | supervised (분류) vs unsupervised (분산) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 한 클래스가 매우 적은 점만 가지면 ($n_k$ 작음) Fisher LDA의 어떤 부분이 불안정해지는가?

<details>
<summary>힌트 및 해설</summary>

$n_k$ 작음 → 그 클래스의 within-class scatter contribution 작음 → $S_W$ 정확한 추정 어려움.

$\mu_k$ 추정도 부정확 → between-class scatter contribution 부정확.

→ 일반화 eigenvalue가 noisy → 잘못된 방향 학습.

해결: Class weighting, class oversampling, 또는 regularized LDA.

</details>

**문제 2** (심화): $S_W$가 singular ($p \gg n$ regime)면 일반화 eigenvalue 문제 풀 수 없다. 어떤 변형이 가능한가?

<details>
<summary>힌트 및 해설</summary>

**Regularized LDA**: $S_W \to S_W + \lambda I$ — Ridge-style 정규화.

**PCA + LDA**: 먼저 PCA로 차원 축소 ($p \to p' < n$) → 그 위에 LDA. 표준 face recognition 파이프라인.

**Pseudoinverse**: $S_W^+ S_B$의 가장 큰 eigenvalue. Min-norm 해.

**Sparse LDA**: $\ell_1$ penalty 추가 — sparse direction 학습.

sklearn `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` — 자동 regularization.

</details>

**문제 3** (ML 연결): Deep Learning의 **t-SNE / UMAP**이 LDA의 비선형 supervised 일반화로 볼 수 있는가? Contrastive learning과의 연결?

<details>
<summary>힌트 및 해설</summary>

**t-SNE**: unsupervised — 레이블 무시. LDA의 supervised 정신은 없음.

**UMAP with target**: `target_metric` 옵션으로 supervised 가능 → 비선형 LDA 같은 효과.

**Contrastive learning** (SimCLR, MoCo): 같은 사진의 augmentation은 가깝게, 다른 사진은 멀게 — 본질적으로 **between/within ratio 최대화의 NN 버전**. 

특히 SupCon (Khosla 2020): 명시적으로 같은 라벨끼리 가깝게, 다른 라벨끼리 멀게 → **NN supervised LDA**.

**결론**: LDA의 핵심 정신 ("같은 클래스 가깝게, 다른 클래스 멀게")이 modern self-supervised + supervised contrastive로 진화. 1936년 Fisher의 통찰이 2020년 NN 학습의 기반.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. GNB vs LDA vs QDA](./02-gnb-lda-qda.md) | [📚 README](../README.md) | [04. Generative vs Discriminative ▶](./04-generative-vs-discriminative.md) |

</div>
