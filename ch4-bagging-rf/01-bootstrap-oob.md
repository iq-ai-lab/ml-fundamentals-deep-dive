# 01. Bootstrap과 OOB Error

## 🎯 핵심 질문

- 크기 $n$의 데이터에서 replacement로 $n$번 뽑은 부트스트랩 샘플은 왜 정확히 **$1 - (1 - 1/n)^n \to 1 - 1/e \approx 63.2\%$** 의 unique 데이터를 포함하는가?
- 남은 **OOB (Out-of-Bag) 샘플**은 어떻게 validation set 역할을 하는가? 별도 hold-out 없이 일반화 오차를 추정할 수 있는가?
- OOB error는 K-fold cross-validation과 어떻게 비교되는가?
- **OOB는 공짜 validation인가** — 편향과 분산은 어떻게 되나?

---

## 🔍 왜 이 개념이 ML에서 중요한가

Bootstrap은 (a) **Bagging의 필수 구성요소** — 트리들에 서로 다른 데이터를 공급, (b) **RF와 GBM에서 OOB로 validation 대체** — train/val split을 안 해도 됨, (c) **통계학의 일반 도구** — 분포의 sample을 재샘플링해 신뢰구간 계산 (Efron 1979), (d) ML과 고전 통계를 잇는 다리. 본 문서는 RF의 `oob_score=True` 파라미터가 왜 유효한 validation 대체인지, 그리고 그 한계가 무엇인지 수학적으로 설명한다.

---

## 📐 수학적 선행 조건

- 이항분포, 극한 $\lim (1 - 1/n)^n = 1/e$
- Cross-validation 개념
- Bias-Variance (Ch1-06)

---

## 📖 직관적 이해

### 부트스트랩의 정의

원 데이터 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$에서 **복원 추출** (with replacement)로 $n$번 뽑은 샘플 $\mathcal{D}^*$. 크기는 $n$로 같지만 중복 있음.

### 왜 63.2%?

한 점 $i$가 한 번 뽑힐 확률 = $1/n$. **뽑히지 않을 확률** $(1 - 1/n)^n \to 1/e \approx 0.368$. 따라서 **뽑히는 확률** $1 - 1/e \approx 0.632$ (= 63.2%).

즉 부트스트랩 샘플은 **원 데이터의 약 63.2%를 포함** (나머지 36.8%는 OOB).

### OOB Error

샘플 $i$가 OOB인 트리들만 사용해 $i$를 예측 → OOB error. 이는 **그 샘플에 대해 학습을 본 적 없는 트리들의 집단 예측** — validation set의 역할.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Bootstrap Sample

원 데이터 $\mathcal{D}_n = \{z_1, \ldots, z_n\}$에서 각 $z_i^*$를 균등 분포 $\{z_1, \ldots, z_n\}$에서 복원 추출로 뽑아 구성한 $\mathcal{D}_n^* = \{z_1^*, \ldots, z_n^*\}$.

### 정의 1.2 — OOB Set

$\mathcal{D}_n^{(\text{oob})} := \mathcal{D}_n \setminus \mathcal{D}_n^*$ — 한 번도 뽑히지 않은 샘플.

### 정의 1.3 — OOB Prediction and Error

$B$개 부트스트랩 샘플 $\{\mathcal{D}^{(b)}\}_{b=1}^B$ 각각으로 모델 $f_b$ 학습. 샘플 $i$의 OOB prediction:

$$\hat{f}^{\text{oob}}(x_i) := \text{avg/majority}\bigl\{f_b(x_i) : i \notin \mathcal{D}^{(b)}\bigr\}.$$

OOB error:

$$\text{Err}^{\text{oob}} := \frac{1}{n}\sum_{i=1}^n L(y_i, \hat{f}^{\text{oob}}(x_i)).$$

---

## 🔬 정리와 증명

### 정리 1.1 — 63.2% Containment

**명제**: $n$-크기 bootstrap에서 한 점이 포함될 확률은 $1 - (1 - 1/n)^n$. $n \to \infty$에서 $1 - 1/e \approx 0.632$.

**증명**: 한 추출에서 점 $i$ 안 뽑힐 확률 = $(n-1)/n = 1 - 1/n$. 독립 $n$회 → 모두 안 뽑힐 확률 = $(1 - 1/n)^n$. 뽑힐 확률 = $1 - (1 - 1/n)^n$.

극한: $\lim_{n \to \infty} (1 - 1/n)^n = e^{-1}$ (유명한 극한). 따라서 $1 - 1/e \approx 0.6321$. $\square$

> 💡 **작은 $n$에서도 유사**: $n=10$이면 $1 - 0.9^{10} \approx 0.651$. $n = 100$이면 $\approx 0.634$. 매우 빨리 63.2%에 수렴.

### 정리 1.2 — 각 트리당 OOB 샘플 수의 분포

**명제**: 부트스트랩 하나당 OOB 샘플 수 $N_{\text{oob}} \sim $ approximately $\text{Binomial}(n, 1/e)$. 특히 $\mathbb{E}[N_{\text{oob}}] \approx n/e \approx 0.368 n$.

**증명 스케치**: 각 $i$가 OOB일 사건은 (근사적으로) 독립·동등 — $1/e$ 확률. 이항 근사. $\square$

### 정리 1.3 — 각 샘플이 OOB로 몇 번 등장하는가

**명제**: $B$개 부트스트랩에서 샘플 $i$가 OOB인 횟수 $M_i \sim \text{Binomial}(B, 1/e)$. $B$ 큰 경우 $\mathbb{E}[M_i] = B/e$.

**즉 $B = 100$ 트리 학습 시 각 샘플 평균 $\approx 36.8$개 트리가 그를 OOB로 사용** → OOB prediction이 안정적.

### 정리 1.4 — OOB Error의 편향

**명제**: OOB error는 **$\approx n/e = 0.368n$개 샘플로 학습한 앙상블의 error를 추정**. 따라서 전체 $n$개로 학습한 앙상블보다 성능이 **약간 더 나쁨** → OOB는 slight pessimistic 추정.

**증명 직관**: OOB predictor는 약 $B/e$개 트리만의 평균 (전체 $B$가 아님). 평균 트리 수가 작으면 variance 증가. 또 각 트리가 $\approx 0.632 n$ unique로 학습 → 정보 손실. → OOB error > true test error (약간).

$n \to \infty$에서 이 편향은 $\to 0$. 실무에서 $n > 500$이면 OOB와 cross-validation이 거의 같음.

### 정리 1.5 — OOB Error vs K-fold CV

| 측면 | OOB | K-fold CV |
|------|-----|-----------|
| 계산 비용 | **앙상블 학습 중 공짜** | 별도 K번 학습 |
| 편향 | slight pessimistic (small $n$) | roughly unbiased (K 크면) |
| 분산 | moderate | K 클수록 작음 |
| 데이터 사용 | 모든 점이 training과 validation에 기여 | 같음 (다른 방식) |
| 앙상블 외 | 사용 불가 | 모든 모델 가능 |

**RF 표준**: `oob_score=True`로 OOB 사용 — 빠르고 정확. 대안으로 `cross_val_score(rf, X, y, cv=5)`.

### 정리 1.6 — Bootstrap의 통계적 정당성 (참고)

**명제** (Efron 1979): 부트스트랩 분포 $\hat{F}_n^*$은 true distribution $F$의 sample 근사 — plug-in estimator. $\sqrt{n}(\hat{\theta}^* - \hat{\theta})$의 분포가 $\sqrt{n}(\hat{\theta} - \theta)$의 분포와 점근적으로 같음.

→ **신뢰구간 구성의 기반**. RF OOB error의 신뢰구간도 이로부터 유도 가능.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. 63.2% 포함률 검증 (정리 1.1)
# ─────────────────────────────────────────────
ns = [10, 50, 100, 1000, 10000]
print(f'{"n":>6s} | {"이론 포함률":>12s} | {"경험 포함률":>12s}')
print('-' * 40)
for n in ns:
    theoretical = 1 - (1 - 1/n)**n
    # 경험: 1000회 부트스트랩의 평균 unique 비율
    uniques = []
    for _ in range(1000):
        bs = rng.integers(0, n, size=n)
        uniques.append(len(np.unique(bs)) / n)
    print(f'{n:>6d} | {theoretical:>12.4f} | {np.mean(uniques):>12.4f}')

# ─────────────────────────────────────────────
# 2. 한 샘플이 OOB인 빈도 (정리 1.3)
# ─────────────────────────────────────────────
n = 100
B = 1000
oob_counts = np.zeros(n)   # 각 샘플이 OOB인 트리 수
for b in range(B):
    bs = rng.integers(0, n, size=n)
    in_bag = set(bs)
    for i in range(n):
        if i not in in_bag:
            oob_counts[i] += 1

print(f'\nB = {B} 부트스트랩에서 각 샘플이 OOB인 횟수:')
print(f'  평균: {oob_counts.mean():.2f}  (이론: B/e = {B/np.e:.2f})')
print(f'  표준편차: {oob_counts.std():.2f}')

# ─────────────────────────────────────────────
# 3. OOB error vs CV error 비교 (정리 1.5)
# ─────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target

rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rf.fit(X, y)

cv_scores = cross_val_score(rf, X, y, cv=5)

print(f'\nRandom Forest on Breast Cancer (n = {len(y)}):')
print(f'  OOB accuracy     : {rf.oob_score_:.4f}')
print(f'  5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
print(f'  Train accuracy   : {rf.score(X, y):.4f}  (overfit 때문에 높음)')

# ─────────────────────────────────────────────
# 4. OOB prediction의 동작 확인
# ─────────────────────────────────────────────
# 각 샘플의 OOB prediction 직접 계산
n_trees = 200
preds_oob = np.zeros((len(y), 2))    # (sample, class)
counts_oob = np.zeros(len(y))

from sklearn.tree import DecisionTreeClassifier
for b in range(n_trees):
    rng_tree = np.random.default_rng(b)
    bs = rng_tree.integers(0, len(y), size=len(y))
    in_bag = set(bs)
    clf = DecisionTreeClassifier(random_state=b).fit(X[bs], y[bs])
    # OOB 샘플 예측
    oob_idx = [i for i in range(len(y)) if i not in in_bag]
    if len(oob_idx) > 0:
        probs = clf.predict_proba(X[oob_idx])
        for j, i in enumerate(oob_idx):
            preds_oob[i] += probs[j]
            counts_oob[i] += 1

# Normalize
mask = counts_oob > 0
final_preds = np.argmax(preds_oob[mask] / counts_oob[mask, None], axis=1)
oob_acc = (final_preds == y[mask]).mean()
print(f'\n직접 구현 OOB accuracy: {oob_acc:.4f}')
print(f'  (sklearn OOB : {rf.oob_score_:.4f} — 약간 다름: random state 차이)')

# ─────────────────────────────────────────────
# 5. Bootstrap 분포로 표본평균의 표준오차 추정
# ─────────────────────────────────────────────
# 본 문서와 직접 관련 — 통계학적 부트스트랩
true_mean = 5.0
data_sample = rng.standard_normal(50) + true_mean

B = 10000
boot_means = [rng.choice(data_sample, size=len(data_sample), replace=True).mean() for _ in range(B)]
print(f'\n표본평균 Bootstrap SE:')
print(f'  원 표본평균        : {data_sample.mean():.4f}')
print(f'  Bootstrap SE       : {np.std(boot_means):.4f}')
print(f'  이론 SE (σ/√n)    : {1.0/np.sqrt(50):.4f}')
```

**출력 예시**:
```
     n |   이론 포함률 |   경험 포함률
----------------------------------------
    10 |       0.6513 |       0.6510
    50 |       0.6358 |       0.6355
   100 |       0.6340 |       0.6345
  1000 |       0.6323 |       0.6323
 10000 |       0.6321 |       0.6321

B = 1000 부트스트랩에서 각 샘플이 OOB인 횟수:
  평균: 368.30  (이론: B/e = 367.88)
  표준편차: 15.12

Random Forest on Breast Cancer (n = 569):
  OOB accuracy     : 0.9631
  5-fold CV accuracy: 0.9648 ± 0.0127
  Train accuracy   : 1.0000  (overfit 때문에 높음)

직접 구현 OOB accuracy: 0.9613
  (sklearn OOB : 0.9631 — 약간 다름: random state 차이)
```

---

## 🔗 실전 활용

- **sklearn `RandomForestClassifier/Regressor(oob_score=True)`**: `rf.oob_score_` 속성으로 접근.
- **빠른 모델 선택**: n_estimators·max_depth 튜닝 시 OOB로 몇 초 만에 비교.
- **Confidence intervals**: bootstrap 재샘플링으로 파라미터 추정량의 CI.
- **통계학의 bootstrap**: mean·median·regression coef의 SE 계산. `scipy.stats.bootstrap`.
- **Stratified bootstrap**: imbalanced data에서 class 비율 유지하며 재샘플링.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| OOB가 모든 모델에 가능 X | 앙상블·bootstrap 기반 모델만 |
| Small $n$에서 편향 | $n < 100$이면 OOB와 CV 차이 커질 수 있음 |
| Sequential data | 시계열 등 i.i.d. 위반 상황에서 부트스트랩 부적합 |
| Rare classes | 매우 imbalanced면 minority class가 OOB에 적음 — stratified 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{Bootstrap contains } 1 - (1 - 1/n)^n \to 1 - 1/e \approx 0.632 \text{ unique samples; OOB }\approx \text{validation}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **63.2% 포함** | $n$ → ∞ 극한의 $1 - 1/e$ |
| **36.8% OOB** | $\approx n/e$ 샘플이 validation 역할 |
| **각 샘플 OOB 빈도** | $\approx B/e$개 트리 |
| **OOB ≈ K-fold CV** | 큰 $n$에서 거의 같음, 무료 |
| **Slight pessimistic bias** | 본질적 — 그러나 작음 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $n = 5$일 때 부트스트랩 하나당 unique 샘플 수의 평균을 계산하라.

<details>
<summary>힌트 및 해설</summary>

샘플 $i$가 부트스트랩에 포함될 확률 $= 1 - (4/5)^5 = 1 - 0.3277 = 0.6723$. 평균 unique 수 = $5 \times 0.6723 = 3.36$. 즉 평균 약 3.4개 unique.

작은 $n$에서는 비율이 63.2%에서 약간 높음.

</details>

**문제 2** (심화): OOB 예측이 사용 가능한 트리 수가 샘플별로 다른 점이 OOB error에 어떤 영향을 주는가? 어떤 샘플의 OOB 예측이 더 신뢰 가능한가?

<details>
<summary>힌트 및 해설</summary>

$M_i \sim \text{Bin}(B, 1/e)$, $\mathbb{E}[M_i] \approx B/e$. 표준편차 $\sqrt{B \cdot 1/e \cdot (1 - 1/e)} \approx 0.48 \sqrt{B}$.

$B = 100$: 평균 $\approx 37$, sd $\approx 4.8$. 대부분 샘플 30~44개 트리로 OOB 예측.

$B = 10$: 평균 $\approx 3.7$, sd $\approx 1.5$. 어떤 샘플은 **1~2개 트리로만** OOB 예측 → 예측 매우 불안정. sklearn은 이런 경우 경고 발행.

**결론**: $B$가 작으면 OOB error의 variance 큼. 실무에서 $B \geq 100$은 되어야 OOB 신뢰 가능.

</details>

**문제 3** (ML 연결): NN을 **Dropout**과 **Ensemble**로 비교할 때, Dropout이 사실 OOB-like 구조를 갖는다는 말의 의미는?

<details>
<summary>힌트 및 해설</summary>

Dropout: 각 배치마다 일부 뉴런을 random하게 "zeroed out". 각 뉴런은 "참여"하는 mini-NN과 "제외"하는 NN이 번갈아 등장.

이는 **implicit ensemble** — 지수적으로 많은 sub-network의 평균으로 볼 수 있음 (Srivastava 2014). 각 sub-network가 다른 뉴런 subset을 "OOB"로 보는 구조.

RF OOB: 각 트리가 일부 샘플을 OOB로.
Dropout: 각 forward pass가 일부 뉴런을 OOB로.

**대응**: 둘 다 "학습 중 일부를 생략 → 나머지로 일반화 능력 향상". RF는 **sample-level** randomness, Dropout은 **neuron-level** randomness. 정신은 같음 — "diversity 유도로 variance 감소."

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch3-05. 결정트리의 한계](../ch3-decision-tree/05-tree-limitations.md) | [📚 README](../README.md) | [02. Bagging의 분산 감소 ▶](./02-bagging-variance-reduction.md) |

</div>
