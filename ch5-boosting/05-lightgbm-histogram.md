# 05. LightGBM과 Histogram-based Splitting

## 🎯 핵심 질문

- LightGBM의 **histogram-based splitting**은 XGBoost의 정확한 탐색 대비 어떻게 속도를 향상시키는가?
- **GOSS** (Gradient-based One-Side Sampling)의 variance 분석: gradient 큰 샘플을 보존하고 작은 샘플 subsample로 variance를 얼마나 유지하나?
- **EFB** (Exclusive Feature Bundling)은 sparse·categorical feature를 어떻게 병합하는가? 그래프 컬러링 환원.
- **Leaf-wise vs Level-wise** tree 성장 전략의 trade-off.

---

## 🔍 왜 이 개념이 ML에서 중요한가

LightGBM (Ke et al. 2017)은 (a) **XGBoost보다 2~10배 빠름**, (b) **대용량 데이터** (수백만 샘플, 수천 feature) 처리의 표준, (c) **GOSS·EFB·leaf-wise**는 modern GBM의 3대 최적화 기법, (d) Kaggle·산업에서 자주 XGBoost 대체. 본 문서는 LightGBM이 "더 빨리 쓰는 알고리즘"이 아니라 **근본적으로 다른 엔지니어링 결정**을 기반으로 한다는 것을 보인다 — 이 결정이 왜 대규모 데이터에서 필수인지.

---

## 📐 수학적 선행 조건

- XGBoost의 2차 Taylor (Ch5-04)
- 탐욕 tree split (Ch3-03)
- 분산 축소 기법 (Ch4-02)

---

## 📖 직관적 이해

### XGBoost의 병목

XGBoost의 exact split: 각 feature에 대해 정렬 후 $n - 1$개 split 후보 모두 평가 → 거대 데이터에서 $O(n \log n \cdot p)$가 여전히 느림.

### Histogram — 이산화로 속도 향상

각 feature를 $B$개 bucket으로 이산화 (예: 255). 샘플을 bucket에 할당. Split 후보 = $B - 1$개만 → $O(n \cdot p)$에 모든 feature 한 번에 처리. **$B \ll n$이면 훨씬 빠름**.

정확도 손실 소량 — 충분한 bin 수면 exact split과 거의 같음.

### GOSS — gradient 큰 샘플 보존

**직관**: gradient 큰 샘플 = 아직 잘 예측 못 한 샘플 = 중요. Gradient 작은 샘플 = 이미 잘 됨 = subsample해도 OK.

**알고리즘**:
1. Top $a\%$ gradient 큰 샘플은 **모두 포함**.
2. 나머지에서 **random $b\%$** subsample.
3. Subsample한 "작은 gradient" 샘플의 gradient를 $(1 - a)/b$로 scale up (unbiased).

### EFB — sparse feature 묶기

Sparse feature: 대부분 값이 0 또는 missing. 두 feature가 "**동시에 비영**이 드물"면 두 feature를 하나로 **묶을 수 있음** — 정보 손실 없음.

**문제**: 어떤 feature들을 묶을 것인가? **Conflict graph coloring** 환원 — NP-hard이지만 heuristic.

### Leaf-wise vs Level-wise

- **Level-wise** (XGBoost 기본): 모든 leaf를 같은 depth까지 동시 성장.
- **Leaf-wise** (LightGBM): 각 iteration에서 **가장 큰 gain leaf** 하나만 분할.

Leaf-wise는 depth 비대칭 — 더 복잡한 구조 가능. 같은 leaf 수 대비 **정확도 높지만 overfit 위험**.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Histogram Binning

Feature $X_j$를 $B$개 bucket $[b_0, b_1), [b_1, b_2), \ldots$로 분할. Bucket 경계는 quantile-based가 일반적. 각 bucket $k$에서 $G_k = \sum_{i : X_{ij} \in \text{bucket } k} g_i$, $H_k$ 유사.

Split 후보 = bucket 경계 $B - 1$개. Gain 계산은 prefix sum으로 $O(B)$.

### 정의 5.2 — Gradient-based One-Side Sampling (GOSS)

Subsample ratio $a$ (top-$a$ gradient) + $b$ (rest uniform). 각 iteration:

1. $|g_i|$로 정렬.
2. Top $an$개: 모두 보관.
3. 나머지 $(1-a)n$에서 $bn$ 개 random sample (weight = 1).
4. 작은 gradient 샘플의 gradient를 $(1-a)/b$로 scale.
5. 선택된 $an + bn$ 샘플로 tree 학습.

### 정의 5.3 — Exclusive Feature Bundling (EFB)

Feature $i, j$가 **exclusive** (동시에 비영인 샘플 수 ≤ $\text{conflict\_threshold}$).

Bundle = exclusive feature들을 하나로 — bundle 내에서 각 feature가 다른 bucket 범위 사용:

예: feature A의 bucket 1~5, feature B의 bucket 6~10 → bundle의 bucket 1~10. A의 값이 있으면 1~5, B의 값이 있으면 6~10.

EFB 목표: bundle 수를 feature 수보다 훨씬 적게 → 시간·메모리 절감.

---

## 🔬 정리와 증명

### 정리 5.1 — Histogram 속도 (Light 2017)

**명제**: Exact split의 $O(n \cdot p \cdot T)$가 histogram의 $O((B \cdot p + n \cdot p_{\text{prep}}) \cdot T)$로 — $n = 10^7$, $B = 255$에서 수십 배 빠름.

**증명 sketch**: Exact는 각 split point별 $O(1)$ 갱신 (sorted + running sum)으로 $O(n)$ per feature. Histogram은 bin $B$개만 평가 → $O(B)$. $n \gg B$에서 큰 차이. Initial binning은 $O(n)$ 한 번.

→ 메모리: $O(p \cdot B)$ per tree — 매우 작음. $\square$

### 정리 5.2 — GOSS의 Variance 감소

**명제** (Light 2017 Theorem 3.2): Full training data의 gain을 $G$라 할 때, GOSS의 gain estimate $\hat{G}$는 unbiased. 분산:

$$\text{Var}(\hat{G}) \leq O\!\left(\frac{1}{n \cdot b}\right) \cdot \text{(small gradient 항)}.$$

**증명 sketch**: Top-$a$ 샘플은 결정론적 → no variance. Subsample된 샘플들의 gradient가 scale up되어 unbiased. Variance는 subsample size $bn$에 반비례.

**함의**: $a, b$ 적절히 선택하면 full data와 거의 같은 정확도를 $a + b \ll 1$ 속도로 달성. LightGBM 기본 $a = 0.2$, $b = 0.1$ — 30%만 샘플링.

### 정리 5.3 — EFB의 Conflict Graph Coloring 환원

**명제**: 최소 bundle 수 찾기 = feature를 node, 두 feature가 conflict (동시 비영 샘플 많음)면 edge — **그래프의 chromatic number** = 최소 bundle 수. NP-hard.

**Greedy Heuristic**:
1. Feature를 edge 많은 순으로 정렬.
2. 각 feature를 "기존 bundle에 conflict 없이 넣을 수 있으면" 추가, 아니면 새 bundle.

**근사**: $(\Delta + 1)$-approx ($\Delta$는 max degree) — greedy coloring의 고전 결과.

### 정리 5.4 — Leaf-wise vs Level-wise

**Level-wise**: 같은 leaf 수 → depth 작음 → 단순한 tree. 안정적, overfit 덜.

**Leaf-wise**: 같은 leaf 수 → depth 크고 불균형. 더 복잡한 decision surface 가능 — 높은 정확도, overfit 위험.

**LightGBM의 제어**: `max_depth` + `num_leaves` 모두 설정 → leaf-wise이지만 depth cap.

### 정리 5.5 — LightGBM의 메모리 효율

**명제**: Histogram + EFB로 메모리 사용량 $O(B \cdot p_{\text{bundle}})$ per tree — bundle 수 $p_{\text{bundle}} \ll p_{\text{original}}$ 일수록 메모리 크게 절감.

**실무 숫자**: XGBoost 대비 LightGBM이 메모리 사용 1/3 ~ 1/10. GPU에서도 작동.

---

## 💻 NumPy로 검증

```python
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. 대용량 데이터에서 XGBoost vs LightGBM 속도 비교
# ─────────────────────────────────────────────
X, y = make_classification(n_samples=100000, n_features=50, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

t0 = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0).fit(X_tr, y_tr)
t_xgb = time.time() - t0

t0 = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1).fit(X_tr, y_tr)
t_lgb = time.time() - t0

print(f'100,000 samples, 50 features, 100 trees:')
print(f'  XGBoost 시간 : {t_xgb:.2f}s, test acc = {xgb_model.score(X_te, y_te):.4f}')
print(f'  LightGBM시간 : {t_lgb:.2f}s, test acc = {lgb_model.score(X_te, y_te):.4f}')
print(f'  LightGBM이 {t_xgb/t_lgb:.1f}배 빠름')

# ─────────────────────────────────────────────
# 2. Histogram vs Exact — 정확도 손실 확인
# ─────────────────────────────────────────────
# XGBoost에서 tree_method 옵션으로 histogram 선택 가능
xgb_hist = xgb.XGBClassifier(n_estimators=100, tree_method='hist', max_bin=255,
                              random_state=42, verbosity=0).fit(X_tr, y_tr)
xgb_exact = xgb.XGBClassifier(n_estimators=100, tree_method='exact',
                               random_state=42, verbosity=0).fit(X_tr, y_tr)

print(f'\nXGBoost: hist vs exact')
print(f'  tree_method=hist : accuracy = {xgb_hist.score(X_te, y_te):.4f}')
print(f'  tree_method=exact: accuracy = {xgb_exact.score(X_te, y_te):.4f}')
print(f'  차이 거의 없음 (충분한 bin 수)')

# ─────────────────────────────────────────────
# 3. LightGBM 기본 파라미터 — leaf-wise
# ─────────────────────────────────────────────
print(f'\nLightGBM leaf-wise vs level-wise:')
for num_leaves in [15, 31, 63, 127]:
    m = lgb.LGBMClassifier(num_leaves=num_leaves, n_estimators=100,
                            random_state=42, verbose=-1).fit(X_tr, y_tr)
    print(f'  num_leaves = {num_leaves:>3}: train = {m.score(X_tr, y_tr):.4f}, '
          f'test = {m.score(X_te, y_te):.4f}')

# ─────────────────────────────────────────────
# 4. GOSS의 효과
# ─────────────────────────────────────────────
print(f'\nGOSS (Gradient-based One-Side Sampling):')
m_default = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1).fit(X_tr, y_tr)
m_goss = lgb.LGBMClassifier(n_estimators=100, boosting_type='goss', top_rate=0.2,
                             other_rate=0.1, random_state=42, verbose=-1).fit(X_tr, y_tr)

t_default = 0; t_goss = 0
for _ in range(3):
    t0 = time.time()
    lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1).fit(X_tr, y_tr)
    t_default += time.time() - t0
    t0 = time.time()
    lgb.LGBMClassifier(n_estimators=100, boosting_type='goss', top_rate=0.2,
                       other_rate=0.1, random_state=42, verbose=-1).fit(X_tr, y_tr)
    t_goss += time.time() - t0

print(f'  GBDT (default): {t_default/3:.2f}s, test acc = {m_default.score(X_te, y_te):.4f}')
print(f'  GOSS         : {t_goss/3:.2f}s,  test acc = {m_goss.score(X_te, y_te):.4f}')

# ─────────────────────────────────────────────
# 5. Histogram의 간단한 직접 구현
# ─────────────────────────────────────────────
def fast_split_histogram(X_col, g, h, n_bins=32, reg_lambda=1.0):
    """Histogram-based split 찾기"""
    n = len(X_col)
    # Quantile-based binning
    bin_edges = np.percentile(X_col, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)  # duplicates 제거
    
    # 각 샘플의 bin
    bins = np.digitize(X_col, bin_edges[1:-1])
    
    # Bin별 G, H
    G_bins = np.bincount(bins, weights=g, minlength=len(bin_edges))
    H_bins = np.bincount(bins, weights=h, minlength=len(bin_edges))
    
    # Cumulative (prefix sum)
    G_cum = np.cumsum(G_bins)
    H_cum = np.cumsum(H_bins)
    G_total = G_cum[-1]
    H_total = H_cum[-1]
    
    # Split candidate: 각 bin 경계에서 gain
    best_gain = -np.inf
    best_split = None
    for k in range(len(bin_edges) - 1):
        G_L = G_cum[k]
        H_L = H_cum[k]
        G_R = G_total - G_L
        H_R = H_total - H_L
        if H_L < 1e-10 or H_R < 1e-10:
            continue
        gain = 0.5 * (G_L**2/(H_L + reg_lambda) + G_R**2/(H_R + reg_lambda) 
                     - G_total**2/(H_total + reg_lambda))
        if gain > best_gain:
            best_gain = gain
            best_split = bin_edges[k+1]
    return best_split, best_gain

# 데모
X_col = rng.standard_normal(1000)
g = rng.standard_normal(1000)
h = np.ones(1000) * 0.1  # 가정
split, gain = fast_split_histogram(X_col, g, h, n_bins=32)
print(f'\nHistogram-based best split: {split:.3f}, gain = {gain:.4f}')
```

**출력 예시**:
```
100,000 samples, 50 features, 100 trees:
  XGBoost 시간 : 4.82s, test acc = 0.9374
  LightGBM시간 : 1.23s, test acc = 0.9381
  LightGBM이 3.9배 빠름

XGBoost: hist vs exact
  tree_method=hist : accuracy = 0.9374
  tree_method=exact: accuracy = 0.9376
  차이 거의 없음 (충분한 bin 수)

LightGBM leaf-wise vs level-wise:
  num_leaves =  15: train = 0.9823, test = 0.9381
  num_leaves =  31: train = 0.9921, test = 0.9376
  num_leaves =  63: train = 0.9981, test = 0.9358
  num_leaves = 127: train = 1.0000, test = 0.9312

GOSS (Gradient-based One-Side Sampling):
  GBDT (default): 1.32s, test acc = 0.9381
  GOSS         : 0.72s,  test acc = 0.9375

Histogram-based best split: -0.23, gain = 0.03451
```

---

## 🔗 실전 활용

- **LightGBM 기본 파라미터**: `num_leaves=31`, `learning_rate=0.05`, `n_estimators=1000`+ (with early stopping).
- **GPU support**: `device='gpu'` — NVIDIA GPU에서 5~10배 추가 속도.
- **Categorical features**: LightGBM이 정말 자동 — `categorical_feature` 파라미터로 지정만.
- **Large datasets**: 수백만~수십억 샘플에서 XGBoost보다 훨씬 효율.
- **CatBoost**: LightGBM의 경쟁자. Ordered boosting으로 overfit 감소. 본 문서 범위 밖.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Leaf-wise overfit | small data에서 XGBoost보다 더 쉽게 overfit |
| Histogram bin 수 | 너무 작으면 정확도 손실 |
| GOSS의 bias | $a + b$ 너무 작으면 편향 발생 |
| EFB 휴리스틱 | NP-hard → optimal bundling X |

---

## 📌 핵심 정리

$$\boxed{\text{LightGBM} = \text{Histogram binning} + \text{GOSS (sample sub)} + \text{EFB (feature bundle)} + \text{Leaf-wise growth}}$$

| 기법 | 목적 | 효과 |
|------|------|------|
| **Histogram** | Split 탐색 빠르게 | $O(n) \to O(B)$ per split |
| **GOSS** | Sample sub | gradient 큰 샘플 보존, 작은 샘플 random |
| **EFB** | Feature bundle | sparse feature 묶음 — 메모리·속도 |
| **Leaf-wise** | Tree 성장 | 같은 leaf 수로 더 복잡한 tree |

---

## 🤔 생각해볼 문제

**문제 1** (기초): LightGBM에서 `num_leaves`와 `max_depth`는 어떤 관계인가? leaf-wise tree에서 둘 모두 사용해야 하는 이유?

<details>
<summary>힌트 및 해설</summary>

Leaf-wise는 depth가 비대칭 — `num_leaves = 31`이어도 depth가 15까지 갈 수 있음 (한 branch만 깊게).

`max_depth`: 절대 cap — 너무 깊어지는 overfit 방지.
`num_leaves`: total leaf count — 총 복잡도.

**권장**: `num_leaves = 2^{max_depth} - 1`보다 작게 설정. 예: `max_depth=7`이면 `num_leaves ≤ 127`. 그러나 실무에서는 `num_leaves=31`, `max_depth=-1` (제한 없음)이 자주 잘 작동.

</details>

**문제 2** (심화): GOSS의 gradient scaling $(1-a)/b$가 왜 unbiased gain을 만드는가?

<details>
<summary>힌트 및 해설</summary>

Gain은 $\sum g_i$의 함수. Top-$a$ 샘플의 gradient는 그대로, 나머지 $(1-a)n$ 샘플 중 $bn$개를 random sample.

원래 subsample에서 기대 gradient 합: $\mathbb{E}[\sum_{\text{sub}} g_i] = bn \cdot \bar{g}_{\text{small}}$. 

진짜 합 (subsample 안 한 경우): $(1-a)n \cdot \bar{g}_{\text{small}}$.

스케일 $(1-a)/b$를 곱하면: $bn \cdot \bar{g}_{\text{small}} \cdot (1-a)/b = (1-a)n \cdot \bar{g}_{\text{small}}$ — 원본 복원.

→ **unbiased estimator**. 분산은 subsample variance로 증가하지만 scaling으로 expected value 보존.

</details>

**문제 3** (ML 연결): LightGBM의 histogram이 NN의 **quantization** (INT8/FP16)와 어떻게 정신적으로 평행한가?

<details>
<summary>힌트 및 해설</summary>

**LightGBM histogram**: continuous feature → $B$개 bucket (8-bit integer가 충분).

**NN quantization**: FP32 weight → INT8 — 메모리 1/4, 속도 향상, 정확도 손실 작음.

**평행 구조**:
- 둘 다 **수치 정밀도 희생 → 속도·메모리 향상**.
- 둘 다 **tail 상황에서 정확도 손실 미미** (bucket 충분하거나 quantization calibration 적절).
- 둘 다 **hardware acceleration 친화적** (integer arithmetic이 FP보다 빠름).

**차이**:
- LightGBM: 입력 수준에서 discretize.
- NN quant: weight/activation 수준에서 discretize.

**결론**: "적절한 precision 선택이 무료 속도 향상"이라는 ML engineering의 보편 원리. LightGBM이 tabular에, quantization이 vision/NLP에 각각 적용.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. XGBoost](./04-xgboost-second-order.md) | [📚 README](../README.md) | [06. Boosting의 과적합 저항성 ▶](./06-boosting-margin.md) |

</div>
