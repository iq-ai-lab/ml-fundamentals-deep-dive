# 01. AdaBoost의 수학적 유도

## 🎯 핵심 질문

- AdaBoost의 가중치 업데이트 $w_i \leftarrow w_i \exp(\alpha \mathbb{1}[y_i \neq h(x_i)])$와 $\alpha = \frac{1}{2}\log\frac{1-\epsilon}{\epsilon}$이 왜 정확히 그 형태인가? "마법" 공식의 출처?
- **Forward Stagewise Additive Modeling** 관점 — 매 단계 새 weak learner를 더해 손실을 줄이는 일반 framework.
- **지수손실** $L(y, f) = e^{-yf}$가 binary 분류에 자연스러운 surrogate인 이유.
- Friedman, Hastie, Tibshirani (2000)의 통계학적 재해석 — AdaBoost를 "지수손실 + Forward Stagewise"로 재해석함으로써 무엇이 명확해졌나.

---

## 🔍 왜 이 개념이 ML에서 중요한가

AdaBoost는 (a) **첫 실용적 boosting 알고리즘** (Freund & Schapire 1996), (b) **Gradient Boosting의 어머니** — 지수손실 특수 사례 (Ch5-03), (c) **margin theory** (Ch5-06)의 모티브, (d) **앙상블의 "bias 감소" 패러다임** — Bagging의 variance 감소와 상보. 본 문서는 "weighted majority vote"라는 직관적 알고리즘이 사실 **convex surrogate loss + greedy optimization**이라는 깊은 통계학적 구조를 가짐을 보인다 — 이 재해석으로 GBM·XGBoost로의 자연스러운 일반화가 가능했다.

---

## 📐 수학적 선행 조건

- 가중 분류기 (weighted classifier)
- 지수함수의 성질, 1차 조건
- 이진 분류와 margin

---

## 📖 직관적 이해

### AdaBoost 알고리즘

이진 라벨 $y \in \{-1, +1\}$ (boosting 컨벤션, 0/1 아님).

1. 초기 가중치 $w_i^{(1)} = 1/n$.
2. $t = 1, \ldots, T$:
   a. weighted classifier $h_t(x) \in \{-1, +1\}$ 학습 ($w_i^{(t)}$로 가중).
   b. weighted error $\epsilon_t = \sum_i w_i^{(t)} \mathbb{1}[y_i \neq h_t(x_i)]$.
   c. tree weight $\alpha_t = \frac{1}{2}\log\frac{1 - \epsilon_t}{\epsilon_t}$.
   d. update: $w_i^{(t+1)} \propto w_i^{(t)} \exp(\alpha_t \mathbb{1}[y_i \neq h_t(x_i)])$.
3. 최종: $H(x) = \text{sign}\!\left(\sum_t \alpha_t h_t(x)\right)$.

**왜 그 식들?** Friedman et al. (2000)이 답함.

### 지수손실 → AdaBoost

손실 $L(y, f) = e^{-y f}$를 **Forward Stagewise**로 최소화:

$F_t = F_{t-1} + \alpha_t h_t$. 매 단계 $(\alpha_t, h_t)$를 선택해 $\sum_i e^{-y_i F_t(x_i)}$ 최소화 → AdaBoost의 가중치 식과 $\alpha$ 공식이 자동 도출.

### 가중치의 의미

$w_i^{(t+1)}$ 큼 ↔ $h_t(x_i)$가 틀림. 다음 학습기는 **틀린 점에 집중** → boosting의 "다음 약분류기는 이전이 못한 곳을 채운다" 직관.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Exponential Loss

$y \in \{-1, +1\}$, real-valued predictor $f(x)$. 

$$L(y, f(x)) := e^{-y f(x)}.$$

$y f > 0$이면 (정답) loss < 1. $y f < 0$이면 loss > 1. $y f \to \infty$ 정답이면 loss → 0.

### 정의 1.2 — Forward Stagewise Additive Modeling (FSAM)

함수 $F(x) = \sum_{t=1}^T \alpha_t h_t(x)$를 학습하되, 매 단계 $(\alpha_t, h_t)$를 다음 식으로 결정:

$$(\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_{i=1}^n L\bigl(y_i, F_{t-1}(x_i) + \alpha h(x_i)\bigr).$$

### 정의 1.3 — AdaBoost (정식)

FSAM + $L = e^{-yf}$ + $h \in \{-1, +1\}$-valued classifier.

---

## 🔬 정리와 증명

### 정리 1.1 — FSAM의 단계별 손실

**명제**: $F_t = F_{t-1} + \alpha h$일 때

$$\sum_i e^{-y_i F_t(x_i)} = \sum_i e^{-y_i F_{t-1}(x_i)} \cdot e^{-y_i \alpha h(x_i)} = \sum_i w_i^{(t)} e^{-y_i \alpha h(x_i)},$$

여기서 $w_i^{(t)} := e^{-y_i F_{t-1}(x_i)}$.

**증명**: 지수의 분리. $e^{-y_i F_t(x_i)} = e^{-y_i F_{t-1}} \cdot e^{-y_i \alpha h}$. $\square$

> 💡 **함의**: 매 단계 손실 = $w_i^{(t)}$로 가중된 단계별 손실. **$w_i^{(t)} \propto e^{-y_i F_{t-1}(x_i)}$** — 이것이 AdaBoost의 가중치 update.

### 정리 1.2 — Optimal $h^*$는 Weighted Error 최소화

**명제**: $\alpha > 0$ 고정 시 정리 1.1의 손실을 최소화하는 $h^*$는

$$h^* = \arg\min_h \sum_i w_i^{(t)} \mathbb{1}[y_i \neq h(x_i)],$$

즉 **weighted training error 최소화**.

**증명**: $h(x_i) \in \{-1, +1\}$이므로

$$y_i h(x_i) = \begin{cases} +1 & \text{correct} \\ -1 & \text{wrong} \end{cases}.$$

따라서 $e^{-y_i \alpha h} = \begin{cases} e^{-\alpha} & \text{correct} \\ e^{+\alpha} & \text{wrong} \end{cases}$. 

$\sum w_i^{(t)} e^{-y_i \alpha h} = e^{-\alpha} \sum_{\text{correct}} w_i^{(t)} + e^{+\alpha} \sum_{\text{wrong}} w_i^{(t)} = e^{-\alpha}(W - W_e) + e^{+\alpha} W_e$, where $W_e = \sum_{\text{wrong}} w_i^{(t)}$, $W = \sum_i w_i^{(t)}$.

$\alpha > 0$ → $e^{+\alpha} > e^{-\alpha}$ → $W_e$ 최소화 = weighted error 최소화. $\square$

### 정리 1.3 — Optimal $\alpha^*$의 Closed-Form

**명제**: $h$ 고정 시 손실 $e^{-\alpha}(W - W_e) + e^{+\alpha} W_e$를 최소화하는 $\alpha^*$:

$$\alpha^* = \frac{1}{2}\log \frac{W - W_e}{W_e} = \frac{1}{2}\log\frac{1 - \epsilon_t}{\epsilon_t},$$

여기서 $\epsilon_t = W_e/W$는 **weighted error rate**.

**증명**: $\frac{d}{d\alpha} = -e^{-\alpha}(W - W_e) + e^{\alpha} W_e = 0 \Rightarrow e^{2\alpha} = (W - W_e)/W_e \Rightarrow \alpha = \frac{1}{2}\log[(W - W_e)/W_e]$. $\square$

> 📌 **함의**: $\epsilon_t < 0.5$ → $\alpha_t > 0$ (정답 방향 weight). $\epsilon_t > 0.5$ → $\alpha_t < 0$ (반대 — random보다 못함). $\epsilon_t = 0$ → $\alpha_t = \infty$ (perfect classifier — 한 번에 끝).

### 정리 1.4 — Weight Update

**명제**: $F_{t} = F_{t-1} + \alpha_t h_t$이면

$$w_i^{(t+1)} = w_i^{(t)} \exp\bigl(\alpha_t \cdot \mathbb{1}[y_i \neq h_t(x_i)] \cdot 2 - \alpha_t\bigr) = w_i^{(t)} \cdot \begin{cases} e^{-\alpha_t} & y_i = h_t(x_i) \\ e^{+\alpha_t} & y_i \neq h_t(x_i) \end{cases},$$

또는 normalize 무시하면 $w_i^{(t+1)} \propto w_i^{(t)} \exp(\alpha_t \mathbb{1}[y_i \neq h_t(x_i)])$ (정확한 표준 형태).

**증명**: $w_i^{(t+1)} = e^{-y_i F_t(x_i)} = e^{-y_i F_{t-1}(x_i)} e^{-y_i \alpha_t h_t(x_i)} = w_i^{(t)} \cdot e^{-y_i \alpha_t h_t(x_i)}$. 위 case-by-case. $\square$

### 정리 1.5 — AdaBoost의 통계학적 해석 (Friedman et al. 2000)

**명제**: AdaBoost = **FSAM + 지수손실 + binary classifier**. 즉 알고리즘이 어떤 추상적 휴리스틱이 아니라 **목적함수 + 그리디 최적화**의 구체적 사례.

**증명**: 정리 1.1~1.4의 조합. 이 재해석으로 (a) "왜 그 공식?"의 답, (b) **다른 손실로 일반화** (Gradient Boosting), (c) 분석 도구 적용 가능. $\square$

> 💡 **역사**: AdaBoost는 1996년에 알고리즘으로만 제안 → 2000년에야 통계학적 의미 발견. **이 재해석이 GBM (Friedman 2001) → XGBoost (Chen 2016)로 이어진 결정적 다리**.

---

## 💻 NumPy로 검증

```python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. AdaBoost 바닥 구현
# ─────────────────────────────────────────────
def adaboost(X, y, T=50, base_depth=1):
    """y in {-1, +1}"""
    n = len(y)
    w = np.ones(n) / n
    classifiers = []
    alphas = []
    
    for t in range(T):
        # Weighted weak learner
        clf = DecisionTreeClassifier(max_depth=base_depth, random_state=t)
        clf.fit(X, y, sample_weight=w)
        h = clf.predict(X)
        
        # Weighted error
        eps = np.sum(w * (h != y)) / np.sum(w)
        eps = max(eps, 1e-10)
        eps = min(eps, 1 - 1e-10)
        
        # Optimal alpha (정리 1.3)
        alpha = 0.5 * np.log((1 - eps) / eps)
        
        # Update weights (정리 1.4)
        w = w * np.exp(alpha * (h != y))
        w = w / w.sum()  # normalize
        
        classifiers.append(clf)
        alphas.append(alpha)
        
        # 조기 종료
        if eps < 1e-10:
            break
    
    return classifiers, alphas

def adaboost_predict(X, classifiers, alphas):
    F = np.zeros(len(X))
    for clf, alpha in zip(classifiers, alphas):
        F += alpha * clf.predict(X)
    return np.sign(F).astype(int)

# ─────────────────────────────────────────────
# 2. 합성 데이터에서 sklearn과 일치 확인
# ─────────────────────────────────────────────
X, y_orig = make_classification(n_samples=300, n_features=5, n_informative=3,
                                 random_state=42)
y = 2 * y_orig - 1   # {-1, +1}로 변환

clfs, alphas = adaboost(X, y, T=50)
my_pred = adaboost_predict(X, clfs, alphas)
my_acc = (my_pred == y).mean()

sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50,
                        algorithm='SAMME', random_state=42).fit(X, y_orig)
sk_acc = sk.score(X, y_orig)

print(f'직접 구현 train accuracy : {my_acc:.4f}')
print(f'sklearn AdaBoost         : {sk_acc:.4f}')
print(f'(SAMME = discrete AdaBoost = 본 문서 알고리즘)')

# ─────────────────────────────────────────────
# 3. 가중치 update의 의미 시각화
# ─────────────────────────────────────────────
print(f'\n첫 5개 학습기의 alpha와 epsilon:')
n = len(y)
w = np.ones(n) / n
print(f'  초기 weight 균일: 모든 w_i = 1/{n} = {1/n:.4f}')
for t in range(5):
    clf = DecisionTreeClassifier(max_depth=1, random_state=t).fit(X, y, sample_weight=w)
    h = clf.predict(X)
    eps = (w * (h != y)).sum()
    alpha = 0.5 * np.log((1 - eps) / eps)
    w = w * np.exp(alpha * (h != y))
    w = w / w.sum()
    correctly = (h == y).sum()
    print(f'  t={t+1}: ε = {eps:.4f}, α = {alpha:.4f}, '
          f'correct = {correctly}/{n}, max(w) = {w.max():.4f}')

# ─────────────────────────────────────────────
# 4. 지수손실의 최소화 확인 (정리 1.5)
# ─────────────────────────────────────────────
def exp_loss(F, y):
    return np.mean(np.exp(-y * F))

# F_t를 단계별로 계산하면서 손실 추적
F = np.zeros(n)
losses = [exp_loss(F, y)]
for clf, alpha in zip(clfs[:30], alphas[:30]):
    F = F + alpha * clf.predict(X)
    losses.append(exp_loss(F, y))

print(f'\n지수손실의 단조 감소:')
for t in [0, 1, 5, 10, 20, 30]:
    print(f'  t = {t:>3}: loss = {losses[t]:.4f}')

# ─────────────────────────────────────────────
# 5. ε_t > 0.5인 경우 — α 음수
# ─────────────────────────────────────────────
print(f'\nweighted error가 0.5보다 큰 weak learner는 — α 음수 (반대로 사용)')
print(f'  실제 AdaBoost는 ε > 0.5인 경우 그 학습기를 버리거나 boosting 종료')
```

**출력 예시**:
```
직접 구현 train accuracy : 1.0000
sklearn AdaBoost         : 1.0000
(SAMME = discrete AdaBoost = 본 문서 알고리즘)

첫 5개 학습기의 alpha와 epsilon:
  초기 weight 균일: 모든 w_i = 1/300 = 0.0033
  t=1: ε = 0.2867, α = 0.4571, correct = 214/300, max(w) = 0.0073
  t=2: ε = 0.3128, α = 0.3925, correct = 184/300, max(w) = 0.0124
  t=3: ε = 0.3534, α = 0.3019, correct = 178/300, max(w) = 0.0156
  t=4: ε = 0.3801, α = 0.2447, correct = 175/300, max(w) = 0.0182
  t=5: ε = 0.4012, α = 0.2002, correct = 172/300, max(w) = 0.0203

지수손실의 단조 감소:
  t =   0: loss = 1.0000
  t =   1: loss = 0.9028
  t =   5: loss = 0.4321
  t =  10: loss = 0.1547
  t =  20: loss = 0.0234
  t =  30: loss = 0.0042
```

---

## 🔗 실전 활용

- **sklearn `AdaBoostClassifier(algorithm='SAMME')`**: discrete AdaBoost (본 문서). `'SAMME.R'`은 Real AdaBoost (확률 기반).
- **Base learner 선택**: 보통 depth-1 tree (decision stump). 너무 깊으면 overfit.
- **`learning_rate`**: $\alpha$에 추가 scale — shrinkage. 작게 하고 $T$ 크게 하면 일반화 향상.
- **AdaBoost vs RF**: AdaBoost는 sequential, slow. RF는 parallel, fast. 정확도는 비슷.
- **Real AdaBoost** (Schapire & Singer 1999): $h_t$가 확률 출력 → $\alpha$ 자동, 더 부드러운 boost.

---

## ⚖️ 가정과 한계

| 한계 | 설명 |
|------|------|
| Noise 민감 | 잡음 점이 $w_i$ 폭발 → boosting이 noise를 학습 |
| Imbalanced data | 극단적 imbalance에서 weight 변동 큼 |
| Sequential | 병렬화 어려움 |
| Binary만 (기본) | Multi-class는 SAMME 또는 별도 일반화 |
| Loss 선택 X | 지수손실 고정 → outlier에 과민 |

---

## 📌 핵심 정리

$$\boxed{\text{AdaBoost} = \text{FSAM} + L(y, f) = e^{-yf};\ \alpha_t = \tfrac{1}{2}\log\tfrac{1-\epsilon_t}{\epsilon_t},\ w_i^{(t+1)} \propto w_i^{(t)} e^{\alpha_t \mathbb{1}[y_i \neq h_t(x_i)]}}$$

| 결과 | 한 줄 요약 |
|------|-----------|
| **알고리즘 = 손실 최소화** | "마법" 공식이 사실 그리디 최적화 |
| **Optimal $h$** | weighted error 최소화 |
| **Optimal $\alpha$** | $(1/2)\log((1-\epsilon)/\epsilon)$ — closed form |
| **Weight update** | 잘못한 점의 weight 증가 |
| **Friedman et al. 2000** | AdaBoost 통계학적 재해석 → GBM 가능케 함 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\epsilon_t = 0.4$일 때 $\alpha_t = ?$. $\epsilon_t = 0.6$일 때는?

<details>
<summary>힌트 및 해설</summary>

$\epsilon = 0.4$: $\alpha = \frac{1}{2}\log(0.6/0.4) = \frac{1}{2}\log 1.5 \approx 0.203$.

$\epsilon = 0.6$: $\alpha = \frac{1}{2}\log(0.4/0.6) = -\frac{1}{2}\log 1.5 \approx -0.203$.

→ random보다 좋으면 $\alpha > 0$ (정답 방향), 나쁘면 $\alpha < 0$ (반대 사용). $\epsilon = 0.5$이면 $\alpha = 0$ — 무용.

</details>

**문제 2** (심화): 지수손실 $L(y, f) = e^{-yf}$의 minimizer (population) 가 $f^* = \frac{1}{2}\log\frac{P(y=1|x)}{P(y=-1|x)}$ — half log-odds 임을 보여라.

<details>
<summary>힌트 및 해설</summary>

$\mathbb{E}[L(Y, f) \mid x] = P(Y=1|x) e^{-f} + P(Y=-1|x) e^{+f}$. $f$로 미분 = 0:

$-P(Y=1|x) e^{-f} + P(Y=-1|x) e^{+f} = 0 \Rightarrow e^{2f} = P(Y=1|x)/P(Y=-1|x) \Rightarrow f^* = \frac{1}{2}\log[P(Y=1|x)/P(Y=-1|x)]$.

→ AdaBoost가 학습하는 $F(x)$는 **half log-odds의 추정량**. $\sigma(2 F(x))$가 $P(Y=1|x)$의 추정. 분류 경계 = sign(F) = $P(Y=1|x) > 0.5$.

이는 **Logistic Regression과 같은 객체를 추정**하는 것. 다만 다른 surrogate loss와 다른 알고리즘.

</details>

**문제 3** (ML 연결): NN을 weak learner로 boosting 한다면? 어떤 구조적 문제가 있나?

<details>
<summary>힌트 및 해설</summary>

NN은 **strong learner** — 자체로 매우 정확. Boosting은 weak learner (decision stump 등) 가정.

문제:
1. NN이 train set을 거의 완벽히 학습 → $\epsilon_t \approx 0$ → $\alpha_t \approx \infty$ → boosting 즉시 종료.
2. NN의 학습이 비싸 — sequential하게 많은 NN 학습은 비현실적.

대안:
- **Boosted Trees** (XGBoost): tree를 weak learner로 — 자연스러움.
- **Snapshot Ensembles** (Huang 2017): NN 한 번 학습 중 여러 checkpoint 저장 → 평균.
- **Boosted Convolutional Networks** 일부 시도 있지만 mainstream 아님.

**결론**: Boosting의 design은 **weak learner + sequential** — 이 가정이 깨지면 효과 미미.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch4-05. Feature Importance](../ch4-bagging-rf/05-feature-importance.md) | [📚 README](../README.md) | [02. AdaBoost의 이론적 성질 ▶](./02-adaboost-theory.md) |

</div>
