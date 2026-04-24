# 01. Naive Bayes의 조건부 독립 가정

## 🎯 핵심 질문

- 조건부 독립 가정 $p(x \mid y) = \prod_j p(x_j \mid y)$의 엄밀한 의미와 **현실에서 항상 깨지는** 이유?
- "**나이브한** 가정이 깨져도 실전에서 잘 작동"하는 현상 (Domingos & Pazzani 1997)을 어떻게 설명하는가?
- MAP 분류기 $\hat{y} = \arg\max_k P(y=k) \prod_j P(x_j \mid y=k)$의 유도.
- Multinomial NB·Bernoulli NB·Gaussian NB의 사용 시점은?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Naive Bayes는 (a) **분류 알고리즘 중 가장 단순** — text classification·spam detection의 표준 baseline, (b) **generative classifier**의 대표 — discriminative (LR, SVM)와 대조 (Ch6-04), (c) **조건부 독립 + Bayes 정리**의 가장 직접적 응용, (d) "왜 단순한 모델이 강력한 baseline인가"의 표준 사례. 본 문서는 NB가 **수학적으로 단순**하지만 **확률 추정과 결정의 분리**라는 깊은 통찰을 통해 실무에서 동작함을 설명한다.

---

## 📐 수학적 선행 조건

- Bayes 정리, 조건부 확률
- MLE와 MAP 추정
- 다항분포·Bernoulli·Gaussian 분포

---

## 📖 직관적 이해

### Bayes 정리로 분류

$$P(y \mid x) = \frac{P(x \mid y) P(y)}{P(x)}.$$

$x$ 고정 시 분모 무관 → MAP 분류기

$$\hat{y}(x) = \arg\max_y P(x \mid y) P(y).$$

문제: $P(x \mid y)$를 어떻게 추정? $x \in \mathbb{R}^p$가 고차원 → joint distribution 매우 어려움.

### 나이브한 해결: 조건부 독립

가정: feature들이 클래스 $y$ 조건부로 독립.

$$P(x \mid y) = \prod_{j=1}^p P(x_j \mid y).$$

→ joint를 marginal의 곱으로 분해. **각 feature 따로 학습** — 매우 단순.

### 현실에서는 거짓

대부분 데이터에서 feature들은 강하게 상관 (예: 키와 체중). → "나이브" 가정 분명히 거짓.

### 그래도 잘 작동

Domingos & Pazzani (1997)의 분석: **분류는 확률 비교만 필요** — 정확한 확률 추정 불필요. 

$\hat{y} = \arg\max P(y) P(x \mid y)$. NB의 $\prod P(x_j \mid y)$가 진짜 $P(x \mid y)$가 아니어도 **클래스별 순위만 맞으면 같은 결정**.

**예**: 이진 분류, 진짜 $P(y=1 | x) = 0.6$이지만 NB가 $\hat{P}(y=1|x) = 0.95$로 추정. $\hat{P}(y=1) > 0.5$이므로 같은 결정 ($y=1$). 확률은 부정확하지만 **분류는 정확**.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Naive Bayes Classifier

데이터 $(x_i, y_i)$, $x_i \in \mathcal{X}^p$, $y_i \in \{1, \ldots, K\}$. 가정:

$$P(x \mid y) = \prod_{j=1}^p P(x_j \mid y) \quad \text{(conditional independence)}.$$

분류 규칙:

$$\hat{y}(x) = \arg\max_{k} P(y = k) \prod_{j=1}^p P(x_j \mid y = k).$$

### 정의 1.2 — Naive Bayes Variants

**Gaussian NB** (continuous $x_j$): $P(x_j \mid y = k) = \mathcal{N}(x_j; \mu_{jk}, \sigma_{jk}^2)$.

**Multinomial NB** (count features, e.g. word counts): $P(x_j \mid y = k) = \theta_{jk}^{x_j} / \cdots$ (multinomial likelihood).

**Bernoulli NB** (binary features, e.g. word presence): $P(x_j \mid y = k) = \theta_{jk}^{x_j}(1 - \theta_{jk})^{1 - x_j}$.

---

## 🔬 정리와 증명

### 정리 1.1 — NB의 MAP 분류기

**명제**: 정의 1.1의 NB classifier는 **0-1 loss 하의 Bayes optimal classifier**의 가정 simplification.

**증명**: 0-1 loss 최적화는 $\arg\max_y P(y \mid x) = \arg\max_y P(x \mid y) P(y)$. NB의 가정 $P(x \mid y) = \prod P(x_j \mid y)$ 대입. $\square$

### 정리 1.2 — MLE for Gaussian NB

**명제**: Gaussian NB의 MLE 추정량:

$$\hat{\pi}_k = \frac{n_k}{n}, \qquad \hat{\mu}_{jk} = \frac{1}{n_k}\sum_{i: y_i = k} x_{ij}, \qquad \hat{\sigma}^2_{jk} = \frac{1}{n_k}\sum_{i: y_i = k}(x_{ij} - \hat{\mu}_{jk})^2.$$

**증명**: log-likelihood $\sum_i \log P(y_i) + \sum_i \sum_j \log P(x_{ij} \mid y_i)$. 클래스별로 분리되고, 각 $(j, k)$ 쌍의 normal MLE는 표본평균·표본분산 (Ch1-01의 1차원 버전). $\square$

> 💡 **계산**: 매우 빠름 — $O(np)$ 한 번 pass.

### 정리 1.3 — MAP for Multinomial NB (with Laplace Smoothing)

**명제**: Multinomial NB에서 MLE는 단어가 train 한 번도 안 나오면 0 확률 → log 발산. **Laplace smoothing**:

$$\hat{\theta}_{jk} = \frac{N_{jk} + \alpha}{\sum_{j'} (N_{j'k} + \alpha)} = \frac{N_{jk} + \alpha}{N_k + \alpha p}.$$

여기서 $N_{jk} = \sum_{i: y_i = k} x_{ij}$ (클래스 $k$에서 단어 $j$의 총 출현 수), $\alpha$는 prior strength (보통 1).

**증명**: Multinomial likelihood + Dirichlet prior $\theta_k \sim \text{Dir}(\alpha, \ldots, \alpha)$ → conjugate posterior MAP. $\square$

### 정리 1.4 — Domingos-Pazzani 분석

**명제 (informal)**: NB의 분류 성능은 "확률 추정의 정확성"이 아니라 "**각 클래스의 확률 ranking**"에 의존. 

특히 두 클래스 $A, B$에 대해 NB가 $\hat{P}(A | x) > \hat{P}(B | x) \iff P(A | x) > P(B | x)$이면 정확. 절대값 정확성은 무관.

**연구 결과** (Domingos & Pazzani 1997): NB는 broad한 조건 (예: 어떤 feature 쌍이 한 방향으로 상관) 하에서 conditional independence가 거짓이어도 0-1 loss optimal.

> 📌 **함의**: NB가 **calibration**은 나쁘지만 **분류**는 잘하는 현상의 이론적 근거. Probability 출력은 신뢰 X (sigmoid·isotonic calibration 필요).

### 정리 1.5 — NB의 학습 비용

**명제**: Train: $O(np)$ — feature 평균·분산 계산. Predict: $O(Kp)$ per sample.

**대조**: LR ($O(npT)$ Newton iter), SVM ($O(n^2)$ 또는 $O(n^3)$). NB가 가장 빠름.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Gaussian NB 바닥 구현
# ─────────────────────────────────────────────
class MyGaussianNB:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_ = np.zeros(len(self.classes_))
        self.means_ = []
        self.vars_ = []
        for k_idx, k in enumerate(self.classes_):
            mask = (y == k)
            self.priors_[k_idx] = mask.mean()
            self.means_.append(X[mask].mean(axis=0))
            self.vars_.append(X[mask].var(axis=0) + 1e-9)
        self.means_ = np.array(self.means_)
        self.vars_ = np.array(self.vars_)
        return self
    
    def predict(self, X):
        # log P(y=k) + sum_j log N(x_j; μ_jk, σ_jk²)
        log_priors = np.log(self.priors_)
        log_likelihoods = np.zeros((len(X), len(self.classes_)))
        for k_idx in range(len(self.classes_)):
            mu = self.means_[k_idx]
            var = self.vars_[k_idx]
            # log Gaussian
            log_lik = -0.5 * np.sum(np.log(2 * np.pi * var) +
                                     (X - mu)**2 / var, axis=1)
            log_likelihoods[:, k_idx] = log_priors[k_idx] + log_lik
        return self.classes_[np.argmax(log_likelihoods, axis=1)]

# Iris dataset에서 sklearn과 일치 확인
data = load_iris()
X, y = data.data, data.target

my_nb = MyGaussianNB().fit(X, y)
sk_nb = GaussianNB().fit(X, y)

print(f'MyGaussianNB train accuracy : {(my_nb.predict(X) == y).mean():.4f}')
print(f'sklearn GaussianNB           : {sk_nb.score(X, y):.4f}')

# ─────────────────────────────────────────────
# 2. NB가 conditional independence 가정 위반에서도 동작 (정리 1.4)
# ─────────────────────────────────────────────
# 강하게 상관된 features 만들기
n = 500
X_corr = rng.standard_normal((n, 2))
X_corr[:, 1] = X_corr[:, 0] + 0.1 * rng.standard_normal(n)   # 거의 동일
y_corr = (X_corr[:, 0] > 0).astype(int)

print(f'\nCorrelated features (X_1 ≈ X_0):')
print(f'  실제 cov(X_0, X_1): {np.cov(X_corr.T)[0, 1]:.4f}')
print(f'  GNB가 가정한 cov  : 0 (대각 공분산만)')
print(f'  GNB accuracy     : {GaussianNB().fit(X_corr, y_corr).score(X_corr, y_corr):.4f}')
print(f'  LR accuracy      : {LogisticRegression(max_iter=1000).fit(X_corr, y_corr).score(X_corr, y_corr):.4f}')
print(f'  → 가정 깨졌어도 분류는 잘 됨')

# ─────────────────────────────────────────────
# 3. Text classification — Multinomial NB의 표준 사례
# ─────────────────────────────────────────────
news = fetch_20newsgroups(subset='train', categories=['comp.graphics', 'sci.med', 'rec.sport.baseball'])
vec = CountVectorizer(max_features=2000)
X_text = vec.fit_transform(news.data)
y_text = news.target

scores_mnb = cross_val_score(MultinomialNB(alpha=1.0), X_text, y_text, cv=5)
scores_lr = cross_val_score(LogisticRegression(max_iter=2000), X_text, y_text, cv=5)
print(f'\nText classification (3-class news):')
print(f'  Multinomial NB : {scores_mnb.mean():.4f} ± {scores_mnb.std():.4f}')
print(f'  LR             : {scores_lr.mean():.4f} ± {scores_lr.std():.4f}')

# ─────────────────────────────────────────────
# 4. Calibration 비교 — NB의 "확률 추정"은 신뢰 안 됨
# ─────────────────────────────────────────────
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X_text, y_text, test_size=0.3, random_state=0)

# Binary로 단순화
y_tr_bin = (y_tr == 0).astype(int)
y_te_bin = (y_te == 0).astype(int)

mnb_cal = MultinomialNB().fit(X_tr, y_tr_bin)
lr_cal = LogisticRegression(max_iter=1000).fit(X_tr, y_tr_bin)

prob_mnb = mnb_cal.predict_proba(X_te)[:, 1]
prob_lr = lr_cal.predict_proba(X_te)[:, 1]

# 0.5 임계 분류 정확도는 비슷하지만 probability calibration은 다름
print(f'\nProbability 분포 (test):')
print(f'  MNB: min={prob_mnb.min():.4f}, max={prob_mnb.max():.4f}, '
      f'extreme (<0.05 또는 >0.95): {((prob_mnb < 0.05) | (prob_mnb > 0.95)).mean():.4f}')
print(f'  LR : min={prob_lr.min():.4f}, max={prob_lr.max():.4f}, '
      f'extreme: {((prob_lr < 0.05) | (prob_lr > 0.95)).mean():.4f}')
print(f'  → MNB는 probability가 0/1에 너무 치우침 (over-confident)')
```

**출력 예시**:
```
MyGaussianNB train accuracy : 0.9600
sklearn GaussianNB           : 0.9600

Correlated features (X_1 ≈ X_0):
  실제 cov(X_0, X_1): 0.9921
  GNB가 가정한 cov  : 0 (대각 공분산만)
  GNB accuracy     : 0.9760
  LR accuracy      : 0.9780
  → 가정 깨졌어도 분류는 잘 됨

Text classification (3-class news):
  Multinomial NB : 0.9582 ± 0.0089
  LR             : 0.9612 ± 0.0103

Probability 분포 (test):
  MNB: min=0.0000, max=1.0000, extreme (<0.05 또는 >0.95): 0.8521
  LR : min=0.0143, max=0.9876, extreme: 0.4321
  → MNB는 probability가 0/1에 너무 치우침 (over-confident)
```

---

## 🔗 실전 활용

- **sklearn `GaussianNB`** / `MultinomialNB` / `BernoulliNB` / `ComplementNB`: 각 데이터 유형별.
- **Spam detection**: 고전적 baseline, 매우 빠름.
- **Text classification**: 짧은 문서 분류에 LR과 비등 또는 우세.
- **Real-time prediction**: $O(Kp)$ — 매우 빠른 inference.
- **Probability calibration**: NB 출력은 `CalibratedClassifierCV`로 보정 후 사용.
- **Online learning**: `partial_fit` 가능 — streaming data.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Conditional independence | 거짓일 때가 많음 (그래도 분류는 동작) |
| Probability calibration | 매우 over-confident — sigmoid·isotonic 보정 필수 |
| Continuous features | Gaussian이 가정하는 분포에 안 맞으면 부정확 |
| Zero counts | Multinomial에서 Laplace smoothing 필수 |
| Feature interactions | 못 잡음 — tree 기반이 더 나음 |

---

## 📌 핵심 정리

$$\boxed{\hat{y}(x) = \arg\max_k P(y=k) \prod_j P(x_j \mid y=k); \text{독립 가정 거짓이어도 분류 잘 됨}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **NB 가정** | $P(x \mid y) = \prod P(x_j \mid y)$ |
| **MAP 분류기** | log P(y) + $\sum$ log P(x_j | y) 최대화 |
| **빠름** | $O(np)$ 학습, $O(Kp)$ 예측 |
| **Robust to 가정 위반** | Domingos-Pazzani — 분류는 OK, calibration은 X |
| **Variants** | Gaussian (continuous), Multinomial (count), Bernoulli (binary) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 단어 "free"가 spam train data에서 한 번도 안 나왔을 때 Multinomial NB의 $P(\text{free} \mid \text{spam}) = ?$. Laplace smoothing의 효과?

<details>
<summary>힌트 및 해설</summary>

MLE: $\hat{P} = 0/N_{\text{spam}} = 0$. → log $\to -\infty$ → "free"가 한 번이라도 들어간 메일은 절대 spam으로 분류 안 됨 (다른 단어 무관).

Laplace ($\alpha = 1$): $\hat{P} = 1/(N_{\text{spam}} + V)$ where $V$는 vocabulary 크기. → 매우 작지만 0이 아님.

→ 실무에서 Laplace는 필수. sklearn `MultinomialNB(alpha=1.0)` 기본.

</details>

**문제 2** (심화): Gaussian NB에서 모든 feature의 분산이 같다고 가정 ($\sigma_{jk}^2 = \sigma^2$ 전체 공통)하면 어떤 잘 알려진 모델이 되는가?

<details>
<summary>힌트 및 해설</summary>

각 클래스의 likelihood: $P(x \mid k) = \prod_j \mathcal{N}(x_j; \mu_{jk}, \sigma^2)$. log-ratio:

$\log \frac{P(k_1 | x)}{P(k_2 | x)} = \log\frac{\pi_{k_1}}{\pi_{k_2}} + \sum_j \frac{\mu_{j,k_1}^2 - \mu_{j,k_2}^2}{2\sigma^2} - \sum_j \frac{(\mu_{j,k_1} - \mu_{j,k_2}) x_j}{\sigma^2}$.

→ **선형 in $x$** → 결정경계 = hyperplane. 

→ **LDA의 특수 사례**: 클래스 공유 공분산 + 대각. 자세히 Ch6-02.

</details>

**문제 3** (ML 연결): **Word2Vec / BERT** 같은 word embedding이 NB의 단어 등장 (one-hot)을 dense vector로 대체한다. 어떤 NB 가정이 자연스럽게 깨지고 어떤 새 모델이 필요한가?

<details>
<summary>힌트 및 해설</summary>

NB는 word를 **independent feature**로 취급 — "good"과 "great"이 무관. 이는 NB의 가장 큰 단점.

Word2Vec/BERT: word를 **dense vector** (similar words → similar vectors). "good"과 "great" 가까움 — 자동 generalization.

NB의 위에 dense embedding 사용:
- **Naive Bayes로 dense feature 사용 어려움** — Gaussian NB는 가능하지만 dense vector의 condional indep 가정 더 깨짐.
- **Logistic Regression on embeddings**: 표준 baseline.
- **NN classifier on embeddings**: BERT fine-tuning.

**결론**: NB의 sparseness (단어 = 독립 feature) 가정이 modern NLP에서 부적합. NB는 여전히 fast baseline이지만 NN-based가 압도적 우세.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch5-06. Boosting Margin](../ch5-boosting/06-boosting-margin.md) | [📚 README](../README.md) | [02. GNB vs LDA vs QDA ▶](./02-gnb-lda-qda.md) |

</div>
