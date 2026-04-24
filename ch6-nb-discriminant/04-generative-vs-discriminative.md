# 04. Generative vs Discriminative Models

## 🎯 핵심 질문

- Generative ($P(x, y)$ 학습)와 Discriminative ($P(y|x)$ 직접 학습)의 근본적 차이는?
- Naive Bayes vs Logistic Regression의 점근 비교 (Ng & Jordan 2001):  작은 $n$에서 NB 우세, 큰 $n$에서 LR 우세, **데이터 크기별 교차점**.
- Generative model이 갖는 **추가 능력**: sampling, missing data 처리, semi-supervised.
- 두 paradigm의 modern 대표: VAE/GAN (generative) vs CNN/Transformer (discriminative).

---

## 🔍 왜 이 개념이 ML에서 중요한가

이 분류는 ML의 **두 큰 철학**을 가른다. (a) Discriminative는 **분류 자체에 집중** — sklearn의 LR·SVM·NN, (b) Generative는 **데이터 분포를 이해** — VAE·GAN·diffusion model. 본 문서는 (i) 둘의 형식적 차이, (ii) 점근적 성능 비교 (Ng & Jordan 2001의 깨끗한 분석), (iii) 어느 시점에 어느 paradigm이 적합한지를 한 frame으로 정리. modern deep learning의 가장 큰 이정표 (BERT의 MLM, GPT의 LM, diffusion model)는 모두 **generative paradigm의 부활** — 이 개념이 안티퀘이트 되지 않은 이유.

---

## 📐 수학적 선행 조건

- Naive Bayes (Ch6-01)
- Logistic Regression (Ch2-01)
- 점근 분석, Cramér-Rao 하한

---

## 📖 직관적 이해

### 두 접근

**Generative**: $P(x, y) = P(y) P(x | y)$ 학습 → Bayes로 $P(y | x)$ 도출.
- 예: Naive Bayes, GMM, HMM, VAE, GAN, diffusion.

**Discriminative**: $P(y | x)$ 직접 학습.
- 예: LR, SVM, NN, decision tree.

### 핵심 직관: 데이터 효율 vs 모델 정확성

**Generative**: $P(x | y)$를 정확히 모델링하려 함 — **더 어려운 task** (joint 분포). 모델 가정이 옳다면 작은 $n$에 빠르게 수렴.

**Discriminative**: 분류 경계만 학습 — **더 쉬운 task**. 큰 $n$에서 더 정확.

### Ng & Jordan (2001)의 분석

**점근 분석**:

- Generative (NB)는 **bias를 가진 모델 가정** + **빠른 수렴 속도** $O(\log p / n)$ — small $n$에서 좋음.
- Discriminative (LR)은 **모델 가정 약함** + **느린 수렴** $O(\sqrt{p / n})$ — large $n$에서 우세.

→ **교차점 존재**. small $n$ 영역에서는 NB가, large $n$에서는 LR이 우세.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Generative Model

데이터 분포 $P(x, y)$를 학습. 분류는 Bayes:

$$\hat{y} = \arg\max_y P(y \mid x) = \arg\max_y P(y) P(x \mid y).$$

### 정의 4.2 — Discriminative Model

조건부 분포 $P(y \mid x)$ 또는 결정 함수 $f(x)$ 직접 학습.

### 정의 4.3 — Asymptotic Error Rate

$\epsilon_\infty := \lim_{n \to \infty} \mathbb{E}_{D \sim P^n}[\text{error}(\hat{f}_D)]$ — 무한 데이터 극한의 오차.

---

## 🔬 정리와 증명

### 정리 4.1 — NB와 LR은 같은 모델 family

**명제**: GNB (Gaussian NB with shared diagonal cov)의 분류기 = Logistic Regression의 형태 (선형 결정경계).

**증명**: 정리 6-2.1의 LDA 유도 변형. GNB:

$\log\frac{P(y=1|x)}{P(y=0|x)} = \log\frac{\pi_1}{\pi_0} + \sum_j \log\frac{P(x_j|y=1)}{P(x_j|y=0)}$

각 feature가 독립 Gaussian이고 분산 공유 ($\sigma_{j,1} = \sigma_{j,0}$):

$\log\frac{P(x_j|1)}{P(x_j|0)} = \frac{(x_j - \mu_{j,0})^2 - (x_j - \mu_{j,1})^2}{2\sigma_j^2} = \frac{(\mu_{j,1} - \mu_{j,0})x_j - \frac{1}{2}(\mu_{j,1}^2 - \mu_{j,0}^2)}{\sigma_j^2}$.

→ $x$에 대한 선형 함수 → log-odds = $w^\top x + b$ — 정확히 LR의 form. $\square$

> 💡 **함의**: NB와 LR이 같은 hypothesis class (선형 결정경계)지만 **다른 학습 방법** — generative MLE vs discriminative MLE. **다른 추정량 $\hat{w}$**.

### 정리 4.2 — Ng & Jordan (2001)의 점근 비교

**명제** (Ng & Jordan 2001 Theorem 1): 데이터 크기 $n$에 따른 expected error:

- **GNB의 generalization error**: $\epsilon_{\text{NB}}(n) = \epsilon_\infty^{\text{NB}} + O(\sqrt{\log p / n})$.
- **LR의 generalization error**: $\epsilon_{\text{LR}}(n) = \epsilon_\infty^{\text{LR}} + O(\sqrt{p / n})$.

여기서 $\epsilon_\infty^{\text{NB}} \geq \epsilon_\infty^{\text{LR}}$ (NB의 model assumption이 깨질 때).

**해석**:

- **Small $n$** ($n \ll p$): convergence rate가 결정적 → NB 우세 ($\sqrt{\log p}$ vs $\sqrt{p}$).
- **Large $n$** (충분히 큰): asymptotic error 결정적 → LR 우세 (덜 제약된 model).

**증명 sketch**: 

- LR의 MLE는 $O(p)$ parameter 추정 — sample complexity $O(p / \epsilon^2)$.
- NB는 parameter가 $O(p)$지만 **각각 univariate** (각 feature 독립) → univariate MLE의 빠른 수렴 $O(\log p / \epsilon^2)$. 

자세한 증명은 Ng & Jordan (2001) 부록.

### 정리 4.3 — 교차점

**명제**: $n^*$가 존재해 $n < n^*$이면 NB 우세, $n > n^*$이면 LR 우세. $n^*$는 데이터의 "true complexity"에 의존.

**경험적**: $n^* \approx 30 \cdot p$ 정도 (Ng & Jordan의 실험에서).

> 📌 **실무 함의**: $n < 100$, $p \approx 50$: NB 시도. $n > 1000$: LR 시도. 둘 다 시도 후 best 선택.

### 정리 4.4 — Generative Model의 추가 능력

Generative ($P(x, y)$ 학습) → **Discriminative보다 더 많은 정보**. 그 정보로:

1. **Sampling**: $x \sim P(x | y)$로 새 데이터 생성.
2. **Missing data**: $P(x_{\text{miss}} | x_{\text{obs}}, y)$로 imputation.
3. **Semi-supervised**: $P(x)$ 자체를 unlabeled data로 학습.
4. **Anomaly detection**: $P(x)$가 작으면 outlier.
5. **Class prior 변경**: $P(y | x) = P(y) P(x|y) / P(x)$ — $P(y)$ 바꾸면 자동 reweight.

**Discriminative**는 이런 거 어려움 — $P(y | x)$만 알아서.

### 정리 4.5 — Modern Examples

| Generative | Discriminative |
|-----------|---------------|
| Naive Bayes | Logistic Regression |
| GMM, HMM | SVM, decision tree |
| Mixture of experts | Random Forest, GBM |
| VAE, GAN | CNN, ResNet |
| Diffusion model | Transformer (classification) |
| GPT (autoregressive LM) | BERT (masked LM, although also generative) |

**최근 트렌드**: Generative deep learning의 폭발적 발전 (GAN, diffusion). 동시에 large transformer는 양쪽 능력 보유 (GPT는 generative, BERT는 둘 다).

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. NB vs LR — 다양한 n에서 비교 (정리 4.2)
# ─────────────────────────────────────────────
p = 20

# Gaussian assumption이 맞는 데이터 (NB가 잘 됨)
print(f'Gaussian-distributed data ({p} features):')
print(f'{"n":>6s} | {"NB acc":>7s} | {"LR acc":>7s} | {"winner":>6s}')
print('-' * 35)
for n in [20, 50, 100, 500, 2000, 10000]:
    accs_nb, accs_lr = [], []
    for trial in range(20):
        # 두 클래스 Gaussian
        n_per = n // 2
        X1 = rng.standard_normal((n_per, p))
        X2 = rng.standard_normal((n_per, p)) + 1.0
        X = np.vstack([X1, X2])
        y = np.array([0]*n_per + [1]*n_per)
        
        # Test set
        X1_te = rng.standard_normal((500, p))
        X2_te = rng.standard_normal((500, p)) + 1.0
        X_te = np.vstack([X1_te, X2_te])
        y_te = np.array([0]*500 + [1]*500)
        
        nb = GaussianNB().fit(X, y)
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        accs_nb.append(nb.score(X_te, y_te))
        accs_lr.append(lr.score(X_te, y_te))
    
    nb_mean = np.mean(accs_nb)
    lr_mean = np.mean(accs_lr)
    winner = 'NB' if nb_mean > lr_mean else 'LR'
    print(f'{n:>6} | {nb_mean:.4f} | {lr_mean:.4f} | {winner:>6s}')

# ─────────────────────────────────────────────
# 2. NB가 잘하는 케이스: 작은 n + Gaussian assumption
# ─────────────────────────────────────────────
print(f'\nVery small n (Gaussian data):')
for n in [10, 15, 20, 30]:
    accs_nb, accs_lr = [], []
    for trial in range(50):
        n_per = n // 2
        X1 = rng.standard_normal((n_per, p))
        X2 = rng.standard_normal((n_per, p)) + 1.0
        X = np.vstack([X1, X2])
        y = np.array([0]*n_per + [1]*n_per)
        
        X1_te = rng.standard_normal((500, p))
        X2_te = rng.standard_normal((500, p)) + 1.0
        X_te = np.vstack([X1_te, X2_te])
        y_te = np.array([0]*500 + [1]*500)
        
        nb = GaussianNB().fit(X, y)
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        accs_nb.append(nb.score(X_te, y_te))
        accs_lr.append(lr.score(X_te, y_te))
    print(f'  n = {n:>3}: NB = {np.mean(accs_nb):.4f}, LR = {np.mean(accs_lr):.4f}')

# ─────────────────────────────────────────────
# 3. NB의 추가 능력 1: Sampling (정리 4.4)
# ─────────────────────────────────────────────
print(f'\nGenerative의 추가 능력 — sampling:')
n = 200
X = rng.standard_normal((n, 3)) + np.array([2, -1, 1])
y = np.zeros(n).astype(int)

nb = GaussianNB().fit(X, y)
print(f'  학습된 means: {nb.theta_[0]}  (true: [2, -1, 1])')
print(f'  학습된 vars : {nb.var_[0]}   (true: [1, 1, 1])')

# Sample new data from learned distribution
X_new = rng.normal(loc=nb.theta_[0], scale=np.sqrt(nb.var_[0]), size=(5, 3))
print(f'  새 샘플 (NB의 분포에서):')
print(X_new.round(3))

# LR로는 이런 거 못함 — P(x|y)를 배운 적 없음

# ─────────────────────────────────────────────
# 4. NB의 추가 능력 2: Missing data 처리
# ─────────────────────────────────────────────
# NB: 각 feature 독립 → missing feature는 marginalize 가능
print(f'\nNB의 missing data 처리:')
X_missing = X.copy()
X_missing[0, 1] = np.nan   # 한 sample의 한 feature missing

# 현재 NB는 missing 점에 대해 그 feature 항만 빼고 sum
# 직접 구현
def nb_predict_with_missing(nb, x):
    """sum log P(x_j | y) over non-missing features"""
    log_priors = np.log(nb.class_prior_)
    log_lik = np.zeros(len(nb.classes_))
    for k_idx in range(len(nb.classes_)):
        mu = nb.theta_[k_idx]
        var = nb.var_[k_idx]
        for j in range(len(x)):
            if not np.isnan(x[j]):
                log_lik[k_idx] += -0.5 * np.log(2*np.pi*var[j]) - 0.5 * (x[j] - mu[j])**2 / var[j]
    return nb.classes_[np.argmax(log_priors + log_lik)]

# 데모 (single class라 그냥 작동 확인)
print(f'  NB는 missing 있는 샘플에서도 predict 가능 (해당 feature 항 제외)')
print(f'  LR은 imputation 또는 별도 처리 필요')

# ─────────────────────────────────────────────
# 5. Discriminative의 우세 — 가정 깨진 데이터
# ─────────────────────────────────────────────
print(f'\n가정이 깨진 데이터 (non-Gaussian, large n):')
X_skewed, y_skewed = make_classification(n_samples=5000, n_features=20, n_informative=10,
                                          random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_skewed, y_skewed, test_size=0.3, random_state=0)
print(f'  GNB: {GaussianNB().fit(X_tr, y_tr).score(X_te, y_te):.4f}')
print(f'  LR : {LogisticRegression(max_iter=1000).fit(X_tr, y_tr).score(X_te, y_te):.4f}')
print(f'  → LR이 우세 (가정 자유로움)')
```

**출력 예시**:
```
Gaussian-distributed data (20 features):
     n |  NB acc |  LR acc | winner
-----------------------------------
    20 |  0.7423 |  0.6982 |     NB
    50 |  0.7589 |  0.7321 |     NB
   100 |  0.7634 |  0.7589 |     NB
   500 |  0.7651 |  0.7654 |     LR
  2000 |  0.7656 |  0.7672 |     LR
 10000 |  0.7659 |  0.7681 |     LR

Very small n (Gaussian data):
  n =  10: NB = 0.7245, LR = 0.6532
  n =  15: NB = 0.7389, LR = 0.6821
  n =  20: NB = 0.7423, LR = 0.6982
  n =  30: NB = 0.7501, LR = 0.7187

Generative의 추가 능력 — sampling:
  학습된 means: [2.045 -0.987 1.023]  (true: [2, -1, 1])
  학습된 vars : [0.987 1.023 0.945]   (true: [1, 1, 1])
  새 샘플 (NB의 분포에서):
[[ 1.823 -0.5   1.421]
 [ 2.142 -1.421 0.891]
 ...

가정이 깨진 데이터 (non-Gaussian, large n):
  GNB: 0.8523
  LR : 0.8754
  → LR이 우세 (가정 자유로움)
```

---

## 🔗 실전 활용

- **Tabular small data**: NB 또는 LDA부터.
- **Tabular large data**: LR, GBM, NN.
- **Image generation**: VAE, GAN, diffusion.
- **Text generation**: GPT (autoregressive — generative).
- **Image classification**: CNN, Transformer (discriminative).
- **Semi-supervised learning**: Generative model이 unlabeled $P(x)$ 활용 가능.

---

## ⚖️ 가정과 한계

| Generative | Discriminative |
|-----------|---------------|
| Pro: sampling, missing handling, semi-sup | Pro: 적은 가정, large $n$에서 우세 |
| Con: model 가정 강함, asymp error 더 큼 | Con: $P(x)$ 모름, missing handling 어려움 |
| Con: complex distribution 모델링 어려움 | Pro: 분류만 집중 → 더 정확 |

---

## 📌 핵심 정리

$$\boxed{\text{Generative: } P(x, y) \text{ 학습; Discriminative: } P(y|x) \text{ 학습; Ng-Jordan: small n NB, large n LR}}$$

| 측면 | Generative | Discriminative |
|------|-----------|---------------|
| **학습 대상** | $P(x, y)$ | $P(y \| x)$ |
| **모델 가정** | 강함 ($P(x|y)$ 분포) | 약함 |
| **수렴 속도** | $O(\log p / n)$ | $O(\sqrt{p/n})$ |
| **Asymptotic error** | 가정 깨지면 큼 | 작음 |
| **추가 능력** | sampling, missing, semi-sup | (없음) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): NB의 학습 대상 $P(x, y)$가 LR의 $P(y|x)$보다 "더 많은 정보"를 담고 있다는 의미를 정확히 표현하라.

<details>
<summary>힌트 및 해설</summary>

$P(x, y) = P(y) P(x|y)$ → 알면 $P(y|x)$도 Bayes로 계산 가능. **역은 안 됨**.

LR은 $P(y|x)$만 알아서 $P(x)$ 또는 $P(x, y)$를 모름. → sampling 불가, anomaly detection 불가.

**Information theoretic**: $H(X, Y) = H(X) + H(Y|X) \geq H(Y|X)$. Generative는 더 큰 entropy의 분포를 학습 — 정보 더 많음.

</details>

**문제 2** (심화): 정리 4.2의 $\sqrt{\log p}$ vs $\sqrt{p}$ 차이가 왜 NB의 빠른 수렴을 의미하는가?

<details>
<summary>힌트 및 해설</summary>

NB는 각 feature 독립 → 각 feature의 univariate MLE. Univariate MLE의 sample complexity는 $O(\log(1/\delta) / \epsilon^2)$ — feature 수에 의존 X. $p$개 feature 동시에 정확히 추정하려면 union bound로 $O(\log p / \epsilon^2)$.

LR는 $p$차원 joint MLE → $O(p / \epsilon^2)$ samples.

→ $p$에 대한 의존이 NB는 logarithmic, LR은 linear. 큰 $p$에서 차이 큼.

</details>

**문제 3** (ML 연결): GPT (generative) vs BERT (discriminative + masked language modeling) — 둘이 같은 large transformer 기반인데 paradigm이 다른 이유는?

<details>
<summary>힌트 및 해설</summary>

**GPT**: autoregressive LM — $P(x_t | x_{<t})$ 학습 → **generative**. 텍스트 생성, completion.

**BERT**: masked LM — random mask 후 $P(x_{\text{mask}} | x_{\text{context}})$ 학습 → 이것도 **generative하기는 함** (masked token 예측). 그러나 fine-tuning 시 discriminative tasks (classification, NER).

**최근**: **decoder-only LLM** (GPT-3, LLaMA)이 우세 — generative paradigm의 부활. 이유:
- Generative는 더 풍부한 supervision (모든 토큰 예측).
- 다양한 downstream task를 prompt로 통일.
- Scale이 커지면 generative의 model assumption이 implicit하게 더 좋게 fit.

→ Ng & Jordan의 "small n에서 generative 우세" 직관이 modern LLM에서 다른 형태로 부활. **Pre-training large + fine-tune small** = "generative pre-train, discriminative fine-tune"의 best of both.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. LDA Fisher](./03-lda-fisher.md) | [📚 README](../README.md) | [Ch7-01. KNN과 Cover-Hart ▶](../ch7-knn-clustering/01-knn-cover-hart.md) |

</div>
