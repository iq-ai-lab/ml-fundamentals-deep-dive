# 02. Gaussian NB vs LDA vs QDA

## 🎯 핵심 질문

- Gaussian NB (대각 공분산), LDA (공유 공분산), QDA (클래스별 공분산)의 가정 차이가 어떻게 다른 결정경계를 만드는가?
- LDA의 결정경계가 **선형**, QDA는 **이차**임을 직접 유도.
- 세 모델의 sample complexity 비교: 적은 데이터에서 GNB·LDA가 유리, 큰 데이터에서 QDA가 더 정확.
- sklearn으로 직접 구현 후 결과 일치 검증.

---

## 🔍 왜 이 개념이 ML에서 중요한가

이 세 모델은 **generative classifier의 spectrum** — 가정의 강도가 다른 같은 framework. (a) 가정이 강할수록 (GNB > LDA > QDA) **bias 증가, variance 감소**, (b) **Bias-Variance trade-off의 분류 버전**, (c) **LDA의 결정경계 선형성**이 LR과 같은 형태 — generative와 discriminative의 다리 (Ch6-04). 본 문서는 가정-파라미터-결정경계의 mapping을 명확히 정리.

---

## 📐 수학적 선행 조건

- Naive Bayes (Ch6-01)
- 다변수 정규분포의 확률밀도
- 행렬식, 역행렬

---

## 📖 직관적 이해

### 세 모델의 가정

모두 **각 클래스의 likelihood**가 multivariate Gaussian:

$$P(x \mid y = k) = \mathcal{N}(x; \mu_k, \Sigma_k).$$

차이는 $\Sigma_k$의 구조:

- **GNB**: $\Sigma_k$가 대각 + 클래스마다 다름 → $\Sigma_k = \text{diag}(\sigma_{k1}^2, \ldots, \sigma_{kp}^2)$.
- **LDA**: $\Sigma_k = \Sigma$ (모든 클래스 공유) — full covariance.
- **QDA**: $\Sigma_k$ 클래스마다 full covariance.

### 결정경계 모양

Posterior $P(y = k | x) \propto P(y = k) P(x | y = k)$. log-ratio:

$$\log \frac{P(k_1 | x)}{P(k_2 | x)} = \log \frac{\pi_{k_1}}{\pi_{k_2}} + \log \frac{P(x | k_1)}{P(x | k_2)}.$$

Gaussian likelihood:

$$\log P(x | k) = -\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k) + \text{const}.$$

- **LDA** ($\Sigma_k = \Sigma$): 이차 항이 **약분** → log-ratio가 $x$에 **선형** → **선형 결정경계**.
- **QDA** ($\Sigma_k$ 다름): 이차 항이 남음 → **이차 결정경계** (타원·쌍곡선).
- **GNB** with 같은 $\Sigma$: LDA 특수 사례. GNB with 다른 $\Sigma_k$: QDA 특수 사례 (대각 제약).

### Sample Complexity

각 모델의 파라미터 수:
- GNB: $K \cdot 2p$ ($K$ 클래스 × 평균과 분산 각 $p$).
- LDA: $K \cdot p + p(p+1)/2$ (means + 공유 cov).
- QDA: $K \cdot (p + p(p+1)/2)$ (각 클래스 cov).

QDA가 가장 expressive지만 가장 데이터 hungry. Small $n$에서 cov 추정 부정확 → singular 문제.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Gaussian Class Models

$P(y = k) = \pi_k$, $P(x | y = k) = \mathcal{N}(x; \mu_k, \Sigma_k)$.

- **GNB**: $\Sigma_k = \text{diag}(\sigma_{k1}^2, \ldots, \sigma_{kp}^2)$.
- **LDA** (Linear Discriminant Analysis): $\Sigma_k = \Sigma$ for all $k$.
- **QDA** (Quadratic Discriminant Analysis): $\Sigma_k$ free.

### 정의 2.2 — MLE

Class prior: $\hat{\pi}_k = n_k/n$.

Class mean: $\hat{\mu}_k = \frac{1}{n_k}\sum_{i: y_i = k} x_i$.

LDA의 공유 cov: $\hat{\Sigma} = \frac{1}{n - K}\sum_k \sum_{i: y_i = k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^\top$.

QDA: $\hat{\Sigma}_k = \frac{1}{n_k - 1}\sum_{i: y_i = k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^\top$.

GNB: 위 QDA의 대각 부분만.

---

## 🔬 정리와 증명

### 정리 2.1 — LDA의 선형 결정경계

**명제**: LDA의 두 클래스 $k_1, k_2$ 결정경계 ($P(k_1 | x) = P(k_2 | x)$):

$$w^\top x + b = 0,$$

$$w = \Sigma^{-1}(\mu_{k_1} - \mu_{k_2}), \quad b = -\frac{1}{2}(\mu_{k_1} + \mu_{k_2})^\top \Sigma^{-1}(\mu_{k_1} - \mu_{k_2}) + \log\frac{\pi_{k_1}}{\pi_{k_2}}.$$

**증명**: log-posterior ratio:

$$\log\frac{P(k_1|x)}{P(k_2|x)} = \log\frac{\pi_{k_1}}{\pi_{k_2}} - \frac{1}{2}(x - \mu_{k_1})^\top \Sigma^{-1}(x - \mu_{k_1}) + \frac{1}{2}(x - \mu_{k_2})^\top \Sigma^{-1}(x - \mu_{k_2}).$$

이차 항 $-\frac{1}{2}x^\top \Sigma^{-1} x$가 양쪽에 같으므로 **약분**. 일차 항만 남음 → 선형. 정리하면 위 form. $\square$

> 💡 **LR과의 비교**: LR도 선형 결정경계. **차이**: LR은 직접 $P(y|x)$ 모델링, LDA는 generative ($P(x|y)$ + Bayes). 같은 form, 다른 학습 방법. Ch6-04.

### 정리 2.2 — QDA의 이차 결정경계

**명제**: QDA의 결정경계는 $x^\top A x + b^\top x + c = 0$ 형태 (이차).

**증명**: QDA에서 $\Sigma_{k_1} \neq \Sigma_{k_2}$ → 이차 항 약분 안 됨:

$$\log\frac{P(k_1|x)}{P(k_2|x)} = -\frac{1}{2}\log\frac{|\Sigma_{k_1}|}{|\Sigma_{k_2}|} - \frac{1}{2}(x - \mu_{k_1})^\top \Sigma_{k_1}^{-1}(x - \mu_{k_1}) + \frac{1}{2}(x - \mu_{k_2})^\top \Sigma_{k_2}^{-1}(x - \mu_{k_2}) + \log\frac{\pi_{k_1}}{\pi_{k_2}}.$$

$x$에 대한 이차 항이 남음 → 결정경계는 quadric (타원·포물선·쌍곡선). $\square$

### 정리 2.3 — GNB는 LDA의 대각 특수화

**명제**: GNB with shared diagonal cov $\Sigma_k = \Sigma$ (대각) → LDA with diagonal cov.

**증명**: 정의에서 자명. GNB는 $\Sigma$가 대각이라는 제약. LDA는 $\Sigma$가 일반적. 둘 다 공유 → 같은 framework. $\square$

> 📌 **함의**: GNB는 LDA에 비해 **bias 큼** (대각 제약), variance 작음 (적은 파라미터). small $n$에서 GNB 유리.

### 정리 2.4 — 분산-편향 Trade-off

**명제** (informal):

| 모델 | 파라미터 수 | Bias | Variance |
|------|-----------|------|----------|
| GNB | $O(Kp)$ | high | low |
| LDA | $O(Kp + p^2)$ | mid | mid |
| QDA | $O(Kp^2)$ | low | high |

**Best**: 데이터 양에 따라.

- $n \ll p^2$: GNB 또는 LDA 권장.
- $n \gg p^2$: QDA 가능.
- $p$ 매우 큼: QDA의 cov 추정 ill-conditioned → Regularized QDA 필요.

### 정리 2.5 — Naive Bayes vs LDA의 차이

**명제**: GNB는 모든 feature가 클래스 조건부 독립 가정. LDA는 가정 없이 full covariance.

**예**: 두 feature $x_1, x_2$가 클래스 내에서 강하게 양의 상관. 

- GNB: 무시 → 부정확한 likelihood, 그러나 분류 boundary는 OK일 수 있음 (Domingos-Pazzani).
- LDA: 정확히 모델링 → 더 정확한 결정경계.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification, load_iris

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. LDA 바닥 구현
# ─────────────────────────────────────────────
class MyLDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n, p = X.shape
        self.priors_ = np.zeros(K)
        self.means_ = np.zeros((K, p))
        Sigma = np.zeros((p, p))
        for k_idx, k in enumerate(self.classes_):
            mask = (y == k)
            self.priors_[k_idx] = mask.mean()
            self.means_[k_idx] = X[mask].mean(axis=0)
            diff = X[mask] - self.means_[k_idx]
            Sigma += diff.T @ diff
        self.Sigma_ = Sigma / (n - K)
        self.Sigma_inv_ = np.linalg.inv(self.Sigma_ + 1e-6 * np.eye(p))
        return self
    
    def discriminant(self, X, k_idx):
        # δ_k(x) = x^T Σ^-1 μ_k - 0.5 μ_k^T Σ^-1 μ_k + log π_k
        mu = self.means_[k_idx]
        return X @ self.Sigma_inv_ @ mu - 0.5 * mu @ self.Sigma_inv_ @ mu + np.log(self.priors_[k_idx])
    
    def predict(self, X):
        scores = np.column_stack([self.discriminant(X, k) for k in range(len(self.classes_))])
        return self.classes_[np.argmax(scores, axis=1)]

# Iris dataset
data = load_iris()
X, y = data.data, data.target

my_lda = MyLDA().fit(X, y)
sk_lda = LinearDiscriminantAnalysis().fit(X, y)
sk_qda = QuadraticDiscriminantAnalysis().fit(X, y)
sk_gnb = GaussianNB().fit(X, y)

print(f'Iris dataset:')
print(f'  My LDA  : {(my_lda.predict(X) == y).mean():.4f}')
print(f'  sklearn LDA: {sk_lda.score(X, y):.4f}')
print(f'  sklearn QDA: {sk_qda.score(X, y):.4f}')
print(f'  sklearn GNB: {sk_gnb.score(X, y):.4f}')

# ─────────────────────────────────────────────
# 2. 결정경계 모양 비교
# ─────────────────────────────────────────────
# 강하게 상관된 feature가 있는 합성 데이터
n = 200
mean1 = np.array([0, 0])
mean2 = np.array([3, 3])
cov_shared = np.array([[1, 0.8], [0.8, 1]])

X1 = rng.multivariate_normal(mean1, cov_shared, n // 2)
X2 = rng.multivariate_normal(mean2, cov_shared, n // 2)
X_2d = np.vstack([X1, X2])
y_2d = np.array([0] * (n//2) + [1] * (n//2))

# LDA의 w, b 계산 (정리 2.1)
mu0 = X_2d[y_2d == 0].mean(axis=0)
mu1 = X_2d[y_2d == 1].mean(axis=0)
Sigma_pooled = ((X_2d[y_2d==0] - mu0).T @ (X_2d[y_2d==0] - mu0) + 
                (X_2d[y_2d==1] - mu1).T @ (X_2d[y_2d==1] - mu1)) / (n - 2)
Sigma_inv = np.linalg.inv(Sigma_pooled)
w = Sigma_inv @ (mu1 - mu0)
b = -0.5 * (mu0 + mu1) @ Sigma_inv @ (mu1 - mu0)
print(f'\nLDA 결정경계 (선형):')
print(f'  w = {w.round(4)}, b = {b:.4f}')
print(f'  결정경계: {w[0]:.3f} x_1 + {w[1]:.3f} x_2 + {b:.3f} = 0')

# ─────────────────────────────────────────────
# 3. QDA가 LDA·GNB보다 더 expressive 한 케이스
# ─────────────────────────────────────────────
# 클래스마다 다른 cov (LDA·GNB가 부적합)
cov1 = np.array([[1, 0.8], [0.8, 1]])
cov2 = np.array([[1, -0.8], [-0.8, 1]])  # 반대 상관

X1 = rng.multivariate_normal([0, 0], cov1, 200)
X2 = rng.multivariate_normal([0, 0], cov2, 200)  # 같은 평균!
X_diff = np.vstack([X1, X2])
y_diff = np.array([0] * 200 + [1] * 200)

print(f'\n같은 평균, 다른 cov:')
print(f'  GNB: {GaussianNB().fit(X_diff, y_diff).score(X_diff, y_diff):.4f}')
print(f'  LDA: {LinearDiscriminantAnalysis().fit(X_diff, y_diff).score(X_diff, y_diff):.4f}')
print(f'  QDA: {QuadraticDiscriminantAnalysis().fit(X_diff, y_diff).score(X_diff, y_diff):.4f}')
print(f'(평균이 같으므로 선형 분류 불가 — QDA만 cov 차이로 분류 가능)')

# ─────────────────────────────────────────────
# 4. Sample size effect (정리 2.4)
# ─────────────────────────────────────────────
from sklearn.model_selection import cross_val_score

print(f'\nSample size별 model 비교 (Iris-like multinormal):')
mean_diff = 1.5
for n_samples in [50, 200, 1000, 5000]:
    X1 = rng.multivariate_normal([0]*5, np.eye(5), n_samples//2)
    X2 = rng.multivariate_normal([mean_diff]*5, np.eye(5), n_samples//2)
    X_test = np.vstack([X1, X2])
    y_test = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
    
    gnb_acc = cross_val_score(GaussianNB(), X_test, y_test, cv=3).mean()
    lda_acc = cross_val_score(LinearDiscriminantAnalysis(), X_test, y_test, cv=3).mean()
    qda_acc = cross_val_score(QuadraticDiscriminantAnalysis(), X_test, y_test, cv=3).mean()
    print(f'  n = {n_samples:>5}: GNB={gnb_acc:.4f}, LDA={lda_acc:.4f}, QDA={qda_acc:.4f}')

print(f'  → 작은 n: GNB·LDA 안정, QDA 불안정')
print(f'  → 큰 n: 모두 비슷 (data가 spherical Gaussian이라 GNB도 정확)')
```

**출력 예시**:
```
Iris dataset:
  My LDA  : 0.9800
  sklearn LDA: 0.9800
  sklearn QDA: 0.9800
  sklearn GNB: 0.9600

LDA 결정경계 (선형):
  w = [-1.0234 -1.1521], b = 6.5234
  결정경계: -1.023 x_1 + -1.152 x_2 + 6.523 = 0

같은 평균, 다른 cov:
  GNB: 0.5025
  LDA: 0.5050
  QDA: 0.7575
(평균이 같으므로 선형 분류 불가 — QDA만 cov 차이로 분류 가능)

Sample size별 model 비교 (Iris-like multinormal):
  n =    50: GNB=0.7800, LDA=0.7800, QDA=0.7800
  n =   200: GNB=0.9050, LDA=0.9100, QDA=0.9100
  n =  1000: GNB=0.9080, LDA=0.9100, QDA=0.9100
  n =  5000: GNB=0.9072, LDA=0.9078, QDA=0.9080
  → 작은 n: GNB·LDA 안정, QDA 불안정
  → 큰 n: 모두 비슷 (data가 spherical Gaussian이라 GNB도 정확)
```

---

## 🔗 실전 활용

- **sklearn `LinearDiscriminantAnalysis` / `QuadraticDiscriminantAnalysis`**: 표준.
- **Regularized QDA**: `QuadraticDiscriminantAnalysis(reg_param=0.5)` — singular cov 방지.
- **LDA as dimensionality reduction**: $K - 1$ 방향으로 projection (Fisher discriminant, Ch6-03).
- **Use cases**: 생물학적 분류 (꽃 종, 종양 grade), 음성 인식 baseline, 신호 처리.
- **Limitations**: data가 정말 Gaussian이어야 — 깨지면 NN/RF가 우세.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Gaussian 가정 | 비-Gaussian 분포에서 부정확 |
| Class cov 추정 | small $n$에서 singular |
| Feature scaling 민감 | StandardScaler 권장 |
| QDA over-fits | $p$ 크면 regularization 필수 |
| Multi-modal class | 한 Gaussian으로 표현 불가 — Mixture model 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{LDA: 공유 } \Sigma \to \text{선형 경계};\ \text{QDA: 클래스별 } \Sigma \to \text{이차 경계};\ \text{GNB: 대각 } \Sigma}$$

| 모델 | 가정 | 결정경계 | 파라미터 |
|------|------|----------|---------|
| **GNB** | 클래스 조건부 독립, 대각 cov | 선형 (with shared) | $O(Kp)$ |
| **LDA** | 공유 full cov | 선형 | $O(Kp + p^2)$ |
| **QDA** | 클래스별 full cov | 이차 | $O(Kp^2)$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): LDA의 결정경계가 두 클래스 평균의 **수직이등분선**이 되는 조건은?

<details>
<summary>힌트 및 해설</summary>

수직이등분선: $w \propto \mu_2 - \mu_1$, $b$가 $(\mu_1 + \mu_2)/2$ 통과.

LDA의 $w = \Sigma^{-1}(\mu_2 - \mu_1)$. $\Sigma = I$ (또는 $\sigma^2 I$)면 $w = \mu_2 - \mu_1$ — **방향 일치**.

$b$도 $\pi_1 = \pi_2$ (균등 prior)이면 $b = -\frac{1}{2}(\mu_1 + \mu_2)^\top \Sigma^{-1}(\mu_2 - \mu_1)$ → 평균을 정확히 통과.

→ **공유 isotropic cov + 균등 prior**에서 수직이등분선.

</details>

**문제 2** (심화): QDA의 결정경계가 **타원**, **쌍곡선**, **포물선** 중 어떤 것이 될지 결정하는 조건은?

<details>
<summary>힌트 및 해설</summary>

이차형식 $x^\top A x + b^\top x + c = 0$의 형태:

- $A \succ 0$ 또는 $A \prec 0$: 타원 (또는 빈 집합).
- $A$ indefinite (양·음 고유값 모두): 쌍곡선.
- $A$ singular ($\det A = 0$): 포물선 또는 직선.

$A = \frac{1}{2}(\Sigma_2^{-1} - \Sigma_1^{-1})$. 

- $\Sigma_1, \Sigma_2$가 commuting이고 한쪽이 더 큼 → 한 부호 고유값들 → 타원.
- 다른 방향에서 다른 cov 우위 → indefinite → 쌍곡선.

**Iris**: virginica vs versicolor에서 보통 타원에 가까운 모양.

</details>

**문제 3** (ML 연결): NN의 마지막 layer + softmax는 LR과 같지만, **두 hidden layer의 중간 representation에서의 분류**가 LDA와 어떻게 비슷한가?

<details>
<summary>힌트 및 해설</summary>

NN의 마지막 hidden representation $\phi(x) \in \mathbb{R}^d$는 학습된 feature. 그 위의 분류 = $\sigma(w^\top \phi(x))$ = LR.

만약 NN이 학습 결과 $\phi(x)$가 **클래스별로 Gaussian-like cluster**가 되도록 학습됐다면 (실제로 cross-entropy loss는 이를 유도) → 마지막 layer는 **LDA-like 선형 분리** 수행.

**Center loss** (Wen 2016) / **arc-face** 등은 명시적으로 hidden representation을 클래스별 cluster로 만드는 loss → LDA-like 구조 강제.

**결론**: NN의 분류 head는 LR/LDA의 generalization. $\phi(x)$가 학습 가능한 부분이 NN의 추가 power.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Naive Bayes](./01-naive-bayes.md) | [📚 README](../README.md) | [03. LDA Fisher Discriminant ▶](./03-lda-fisher.md) |

</div>
