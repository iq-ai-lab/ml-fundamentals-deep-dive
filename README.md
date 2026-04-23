<div align="center">

# 📊 ML Fundamentals Deep Dive

**"`sklearn.linear_model.LinearRegression().fit(X, y)`를 호출하는 것과, Normal Equation $\hat{\beta} = (X^\top X)^{-1} X^\top y$를 MLE·기하학적 수직투영·Moore-Penrose pseudoinverse 세 관점에서 유도할 수 있는 것은 다르다"**

<br/>

> *"`RandomForestClassifier`를 **쓰는 것**과, $B \to \infty$에서 RF가 $\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$로 분산이 감소함을 증명할 수 있는 것은 다르다.  
> `XGBoost`를 **쓰는 것**과, Gradient Boosting이 함수공간의 경사하강법이고 AdaBoost의 지수손실 최소화가 그 특수 사례임을 유도할 수 있는 것은 다르다."*

선형 회귀의 세 가지 유도부터 GLM·결정트리·Bagging·Boosting·생성모델·KNN/클러스터링까지  
**"왜 고전 ML이 동작하는가"** 라는 질문으로 sklearn 한 줄 뒤에 숨은 수학을 끝까지 파헤칩니다

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-EB6C2D?style=flat-square)](https://xgboost.ai/)
[![Docs](https://img.shields.io/badge/Docs-36개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-15k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems_proven-210개-success?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-108개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

머신러닝 자료는 대부분 **"`fit` 부르고 `predict` 부르세요"** 에서 멈춥니다. 하지만 왜 $(X^\top X)^{-1}$이 특이할 때 Moore-Penrose pseudoinverse가 자연스러운 대체인지, 왜 sigmoid가 Bernoulli의 **canonical link function**이고 그 결과 Logistic Regression의 log-likelihood가 **concave**가 되는지, AdaBoost의 가중치 업데이트가 왜 **지수손실의 coordinate descent**이고 XGBoost의 leaf 값 $w^* = -G/(H+\lambda)$가 왜 **함수공간 Newton-Raphson 한 스텝**인지 — 이런 "왜"는 어디서도 한 자리에 모이지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "선형 회귀는 $\hat{\beta} = (X^\top X)^{-1} X^\top y$" | **MLE 관점**(Gaussian noise + log-likelihood 최대화), **기하 관점**(잔차 $\perp \text{col}(X)$의 수직투영, $H = X(X^\top X)^{-1} X^\top$의 idempotency), **Pseudoinverse 관점**($X^+ = V \Sigma^+ U^\top$, rank-deficient에서 min-norm 해)을 **한 줄씩 동시에** 유도 |
| "Ridge는 $\lambda \|\beta\|^2$ 페널티" | (1) 정규방정식 $\hat{\beta}_R = (X^\top X + \lambda I)^{-1} X^\top y$, (2) **MAP**($\beta \sim \mathcal{N}(0, \tau^2 I)$ prior), (3) **제약**($\|\beta\| \leq c$)의 쌍대 — 세 해석이 **수치적으로 정확히 같은 식**임을 SVD로 증명 |
| "Lasso는 sparsity를 줍니다" | L1 ball의 **다이아몬드 기하**, Laplace prior MAP, KKT의 subdifferential, **Coordinate Descent**가 왜 자연스러운 알고리즘인지 — soft-thresholding $S_\lambda(z) = \text{sgn}(z)\max(\|z\|-\lambda, 0)$의 closed-form 유도 |
| "Logistic Regression은 sigmoid 분류기" | Bernoulli likelihood로부터 **log-odds = $\beta^\top x$의 필연성**, log-likelihood concavity 증명, Newton-Raphson이 **IRLS와 동치**($H = X^\top W X$), **Exponential Family**의 canonical link로서 sigmoid의 도출 |
| "결정트리는 정보이득으로 분할" | Information Gain이 **상호정보량 $I(Y; A)$와 동치**, Gini가 **분산의 이산 버전**(엔트로피 1차 Taylor 근사), CART의 회귀 트리에서 leaf 평균이 MSE 최소해, **Cost-Complexity Pruning** $R_\alpha(T) = R(T) + \alpha\|T\|$의 weakest link 알고리즘 |
| "Random Forest는 트리 평균" | **Bootstrap의 $1 - 1/e \approx 63.2\%$ 데이터 포함률**과 OOB error, $\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$ 유도, feature subsampling이 왜 $\rho$를 낮춰 **분산을 추가 감소**시키는지 (Breiman 2001) |
| "AdaBoost는 약분류기 가중치 합" | **지수손실 $L(y, f) = e^{-yf}$**의 Forward Stagewise Additive Modeling으로 가중치 업데이트 $w_i \leftarrow w_i e^{\alpha \mathbb{1}[y_i \neq h(x_i)]}$를 **한 줄씩 유도** (Friedman, Hastie, Tibshirani 2000), 훈련 오차의 지수적 감소 $\prod \sqrt{4\epsilon_t(1-\epsilon_t)}$ |
| "XGBoost는 그냥 좋은 부스팅" | Loss의 **2차 Taylor 근사** $L \approx L_{t-1} + g_t \Delta f + \frac{1}{2} h_t \Delta f^2$, leaf 최적값 $w^* = -G/(H + \lambda)$가 **tree 버전 Newton-Raphson 한 스텝**임을 증명, LightGBM의 GOSS·EFB와 비교 |
| "Naive Bayes는 단순한데 잘 됨" | "나이브" 가정이 깨져도 **분류 경계만 정확하면 된다**는 Domingos & Pazzani (1997) 분석, **Generative vs Discriminative**의 점근 비교 (Ng & Jordan 2001) — 작은 $n$에서 NB, 큰 $n$에서 LR이 우세한 교차점 |
| "K-Means는 거리 기반 클러스터링" | **GMM의 hard-assignment 극한**으로서의 K-Means 유도, EM의 일반화로 KMeans 수렴, **K-Means++의 $O(\log k)$ 경쟁비**, Lloyd 알고리즘의 단조감소성 |
| 공식 나열 | NumPy로 Normal Equation·IRLS·CART·Bagging·AdaBoost·Gradient Boosting을 **바닥부터 구현**, sklearn / XGBoost / LightGBM과 **값 단위로 일치 검증** |

---

## 📌 선행 레포 & 후속 레포

```
[Linear Algebra]   ──►   [Probability]   ──►    이 레포    ──►   [NN Theory Deep Dive]
 SVD·Pseudoinverse·     결합/조건부 분포·      ML Fundamentals     선형 회귀 → MLP
 양정치·QR              다변수정규           (LR·GLM·Tree·               │
                                            Ensemble·NB/LDA·             ▼
                                            KNN/Cluster)        [Generalization Theory]
  ▲                       ▲                                       Bias-Var·VC·NTK
  │                       │
[Math Statistics]   [Calculus & Opt]  [Convex Opt]  [Information Theory]
 MLE·Fisher 정보·    Gradient·Newton·  Lagrangian·   상호정보량·엔트로피
 Exponential Family  IRLS              KKT
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Linear Algebra Deep Dive**(Pseudoinverse·SVD·수직투영)와 **Mathematical Statistics Deep Dive**(MLE·Fisher 정보·Exponential Family)를 선행 지식으로 전제합니다. Normal Equation 유도와 GLM의 canonical link 부분에서 매 챕터 활용됩니다.

> 💡 **권장 선행**: 결정트리의 분할 기준에는 [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive)의 상호정보량·엔트로피, Ridge/Lasso의 쌍대 해석에는 [Convex Optimization Deep Dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive)의 Lagrangian·KKT, Bagging/Bias-Variance에는 [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive)의 분산 분해가 도움이 됩니다.

> 🎯 **후속 연결**: 선형 회귀를 "1-layer 신경망"으로, Logistic Regression을 "1-layer + softmax"로 보면 [NN Theory Deep Dive](https://github.com/iq-ai-lab/nn-theory-deep-dive)의 자연스러운 출발점이 됩니다. Bias-Variance와 Boosting의 margin 이론은 [Generalization Theory Deep Dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive)로 이어집니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-선형회귀_3가지_관점-4A90E2?style=for-the-badge)](./ch1-linear-regression/01-mle-derivation.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Logistic·GLM-4A90E2?style=for-the-badge)](./ch2-logistic-glm/01-logistic-mle.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Decision_Tree-4A90E2?style=for-the-badge)](./ch3-decision-tree/01-information-gain.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Bagging·Random_Forest-4A90E2?style=for-the-badge)](./ch4-bagging-rf/01-bootstrap-oob.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Boosting-4A90E2?style=for-the-badge)](./ch5-boosting/01-adaboost-derivation.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Naive_Bayes·LDA-4A90E2?style=for-the-badge)](./ch6-nb-discriminant/01-naive-bayes.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-KNN·Clustering·DR-4A90E2?style=for-the-badge)](./ch7-knn-clustering/01-knn-cover-hart.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 선형 회귀 — 3가지 관점의 유도

> **핵심 질문:** $\hat{\beta} = (X^\top X)^{-1} X^\top y$는 어떻게 MLE·기하학적 수직투영·Moore-Penrose pseudoinverse 세 관점에서 동시에 유도되는가? Ridge의 정규화와 Bayesian prior와 제약 최적화는 왜 같은 해를 내는가? Lasso의 sparsity는 어디서 오는가?

<details>
<summary><b>MLE·기하·Pseudoinverse·Ridge·Lasso·Bias-Variance (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. MLE 관점에서의 선형 회귀](./ch1-linear-regression/01-mle-derivation.md) | $y = X\beta + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2 I)$의 log-likelihood 최대화가 $\min \|y - X\beta\|^2$로 환원, **Normal Equation** $\hat{\beta} = (X^\top X)^{-1} X^\top y$ 유도, MLE 추정량의 비편향성·BLUE(Gauss-Markov) |
| [02. 기하학적 관점 — 수직투영](./ch1-linear-regression/02-geometric-projection.md) | $\hat{y} = X\hat{\beta}$가 $y$의 $\text{col}(X)$로의 수직투영, 잔차 $r = y - \hat{y} \perp \text{col}(X)$에서 정규방정식 기하 유도, **Hat matrix** $H = X(X^\top X)^{-1} X^\top$의 idempotency $H^2 = H$, Hilbert 공간 수직분해 |
| [03. Moore-Penrose Pseudoinverse와 Rank-deficient](./ch1-linear-regression/03-pseudoinverse.md) | $X^\top X$가 특이할 때 $X^+ = \lim_{\lambda \to 0^+} (X^\top X + \lambda I)^{-1} X^\top$, **SVD 표현** $X^+ = V \Sigma^+ U^\top$, **min-norm least-squares 해의 유일성** 증명, QR vs SVD vs Cholesky 수치 비교 |
| [04. Ridge Regression의 3가지 해석](./ch1-linear-regression/04-ridge-three-views.md) | (1) $\hat{\beta}_R = (X^\top X + \lambda I)^{-1} X^\top y$의 정규화, (2) **MAP** ($\beta \sim \mathcal{N}(0, \tau^2 I)$ prior, $\lambda = \sigma^2/\tau^2$), (3) **제약 쌍대** ($\|\beta\| \leq c$의 KKT) — 세 해의 **수치 일치** 검증, SVD로 $\hat{\beta}_R = \sum \frac{\sigma_i}{\sigma_i^2 + \lambda} u_i^\top y \cdot v_i$ |
| [05. Lasso와 Sparsity](./ch1-linear-regression/05-lasso-sparsity.md) | L1 ball의 **다이아몬드 기하**가 왜 sparsity를 강제하는지, Laplace prior MAP, **KKT의 subdifferential** $\partial \|\beta\|_1 \ni \text{sgn}(\beta)$, **Coordinate Descent**의 soft-thresholding $S_\lambda(z)$ 유도, sklearn `Lasso`와 일치 검증 |
| [06. Bias-Variance Decomposition](./ch1-linear-regression/06-bias-variance.md) | $\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Var} + \sigma^2$ **완전 증명**, Ridge가 bias 증가로 variance 감소를 사는 이유, **정규화 경로** ($\lambda$별 계수 궤적), 학습 곡선과 모델 복잡도 |

</details>

<br/>

### 🔹 Chapter 2: Logistic Regression과 일반화선형모형(GLM)

> **핵심 질문:** Logistic Regression의 log-likelihood가 왜 concave인가? Newton-Raphson이 왜 IRLS와 정확히 동치인가? Sigmoid가 왜 Bernoulli의 canonical link function인가? 분리 문제(Separation)는 왜 MLE를 발산시키고 어떻게 해결하는가?

<details>
<summary><b>Logistic MLE부터 분리 문제까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Logistic Regression의 MLE](./ch2-logistic-glm/01-logistic-mle.md) | Bernoulli likelihood $\prod p_i^{y_i}(1-p_i)^{1-y_i}$, **log-odds $\log\frac{p}{1-p} = \beta^\top x$의 필연성**, log-likelihood의 **concavity** 증명 ($\nabla^2 \ell = -X^\top W X \preceq 0$), MLE의 일의성 |
| [02. IRLS — Iteratively Reweighted Least Squares](./ch2-logistic-glm/02-irls.md) | **Newton-Raphson 업데이트가 가중 최소제곱과 동치** 증명, Hessian $H = X^\top W X$ 유도($W = \text{diag}(p_i(1-p_i))$), working response $z_i = \eta_i + (y_i - p_i)/(p_i(1-p_i))$, sklearn/`statsmodels` 결과와 일치 검증 |
| [03. Exponential Family와 Canonical Link](./ch2-logistic-glm/03-exp-family-canonical-link.md) | GLM의 세 구성요소(분포·선형예측자·link function), **canonical link이 왜 MLE를 단순화**(Fisher scoring = IRLS)하는지 증명, Logit/Probit/Log-log/Cloglog 비교, Poisson/Gamma 회귀까지 통일 |
| [04. Multinomial/Softmax Regression](./ch2-logistic-glm/04-softmax-multinomial.md) | One-vs-Rest vs 직접 multinomial, **softmax의 identifiability 문제**(reference class 고정으로 해결), **cross-entropy = categorical MLE** 동치, NumPy로 softmax + Newton 직접 구현 |
| [05. 분리 문제와 Firth Correction](./ch2-logistic-glm/05-separation-firth.md) | 완전분리(complete separation) 시 **MLE 발산** 현상의 기하학적 원인, **Firth의 penalized likelihood** $\ell(\beta) + \frac{1}{2}\log\|I(\beta)\|$로 유한해 보장, Bayesian 관점(Jeffreys prior) |

</details>

<br/>

### 🔹 Chapter 3: 결정트리 (Decision Tree)

> **핵심 질문:** Information Gain은 왜 상호정보량과 동치인가? Gini와 Entropy는 왜 거의 같은 결정을 내리는가? CART의 회귀 트리에서 leaf 평균이 왜 MSE 최소해인가? Cost-Complexity Pruning의 $\alpha$는 어떻게 선택하는가?

<details>
<summary><b>분할 기준부터 가지치기까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 결정트리의 분할 기준 — 정보이득](./ch3-decision-tree/01-information-gain.md) | 엔트로피 감소 $IG(S, A) = H(S) - \sum_v \frac{\|S_v\|}{\|S\|} H(S_v)$가 **상호정보량 $I(Y; A)$와 동치** 증명, ID3 알고리즘 단계별 유도, Information Theory 레포 연결 |
| [02. Gini Impurity vs Entropy](./ch3-decision-tree/02-gini-vs-entropy.md) | Gini $1 - \sum p_k^2$가 **분산의 이산 버전**, **엔트로피의 1차 Taylor 전개**가 Gini와 일치함을 보여 두 기준이 거의 동등한 분할을 내림을 증명, CART가 Gini를 선호하는 계산상 이유 |
| [03. 회귀 트리와 MSE 분할](./ch3-decision-tree/03-regression-tree.md) | leaf 예측값 $c_v = \text{mean}(y \in \text{leaf})$가 **각 leaf 내 MSE 최소해**(1차 미분 = 0), CART의 **탐욕 분할** 알고리즘, 연속 feature의 split point 탐색 $O(n \log n)$, sklearn `DecisionTreeRegressor`와 결과 일치 |
| [04. 가지치기와 Cost-Complexity](./ch3-decision-tree/04-pruning-cost-complexity.md) | $R_\alpha(T) = R(T) + \alpha\|T\|$ 최소화, **Weakest Link Pruning**의 단조 시퀀스 정리, **Cross-validation으로 $\alpha$ 선택**, 1-SE rule, sklearn `cost_complexity_pruning_path`와 비교 |
| [05. 결정트리의 한계 — 불안정성·축정렬 편향](./ch3-decision-tree/05-tree-limitations.md) | 훈련 데이터의 작은 변화가 완전히 다른 트리를 만드는 **high variance** 현상, **축-정렬 분할의 경계 직사각형 편향**(oblique tree로 해결), 이것이 **앙상블의 동기**가 되는 논리 |

</details>

<br/>

### 🔹 Chapter 4: Bagging과 Random Forest

> **핵심 질문:** Bootstrap이 왜 약 63.2%의 데이터를 포함하는가? Bagging의 분산 감소 공식 $\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$는 어디서 오는가? Random Forest의 feature subsampling은 왜 추가 분산 감소를 만드는가? RF가 $B \to \infty$에서 수렴함을 어떻게 증명하는가?

<details>
<summary><b>Bootstrap부터 Feature Importance까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Bootstrap과 OOB Error](./ch4-bagging-rf/01-bootstrap-oob.md) | 부트스트랩 샘플이 $1 - (1 - 1/n)^n \to 1 - 1/e \approx 63.2\%$의 데이터를 포함함의 극한 증명, **OOB 샘플로 validation 대체**, OOB error가 cross-validation과 어떻게 비교되는지 |
| [02. Bagging의 분산 감소 메커니즘](./ch4-bagging-rf/02-bagging-variance-reduction.md) | $B$개 모델 평균의 분산 = $\frac{\sigma^2}{B}$ (독립 가정) → 상관된 경우 **$\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$ 유도**, $B \to \infty$에서 하한이 $\rho\sigma^2$로 결정됨을 보여 "상관관계 감소"가 핵심 레버임을 시사 |
| [03. Random Forest의 추가 무작위성](./ch4-bagging-rf/03-random-forest.md) | 각 split에서 **feature 부분집합 무작위 선택** ($\sqrt{p}$ for classification, $p/3$ for regression)이 tree 간 상관 $\rho$를 낮춰 분산 추가 감소, NumPy로 RF 바닥 구현 후 sklearn과 비교 |
| [04. Random Forest의 수렴성](./ch4-bagging-rf/04-rf-convergence.md) | $B \to \infty$에서 RF predictor가 **무한 앙상블로 수렴**(Breiman 2001), generalization error의 단조감소성, 강한 큰 수의 법칙으로 점근적 정당성, "왜 더 많은 트리는 절대 해롭지 않은가" |
| [05. Feature Importance — Permutation·MDI](./ch4-bagging-rf/05-feature-importance.md) | **Mean Decrease in Impurity (MDI)**의 high-cardinality 편향 증명, **Permutation Importance**가 왜 더 robust한지, **SHAP**로의 일반화, sklearn `permutation_importance`와 비교 |

</details>

<br/>

### 🔹 Chapter 5: Boosting

> **핵심 질문:** AdaBoost의 가중치 업데이트는 왜 정확히 그 형태인가? Gradient Boosting이 왜 함수공간의 경사하강법인가? XGBoost의 leaf 값 $w^* = -G/(H+\lambda)$가 왜 Newton-Raphson 한 스텝인가? AdaBoost는 왜 훈련오차 0 이후에도 테스트 오차가 감소하는가?

<details>
<summary><b>AdaBoost부터 Margin Theory까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. AdaBoost의 수학적 유도](./ch5-boosting/01-adaboost-derivation.md) | **지수손실 $L(y, f) = e^{-yf}$** 가정, Forward Stagewise Additive Modeling으로 **가중치 업데이트** $w_i \leftarrow w_i e^{\alpha \mathbb{1}[y_i \neq h(x_i)]}$와 $\alpha = \frac{1}{2}\log\frac{1-\epsilon}{\epsilon}$를 **한 줄씩 유도** (Friedman, Hastie, Tibshirani 2000) |
| [02. AdaBoost의 이론적 성질](./ch5-boosting/02-adaboost-theory.md) | 훈련 오차의 **지수적 감소** $\prod_t \sqrt{4\epsilon_t(1-\epsilon_t)}$ 증명, margin distribution 관점 (Schapire et al. 1998), VC 경계와 일반화 오차의 관계 |
| [03. Gradient Boosting — 함수공간 경사하강법](./ch5-boosting/03-gradient-boosting.md) | 손실 $L(y, f)$의 **음의 gradient $-\partial L / \partial f$에 약분류기 fit**, $F_t = F_{t-1} + \eta h_t$, 함수공간 $\mathcal{F}$에서의 steepest descent 해석, **AdaBoost가 GBM의 지수손실 특수 사례**임을 증명 |
| [04. XGBoost — 2차 Taylor 근사](./ch5-boosting/04-xgboost-second-order.md) | $L_t \approx L_{t-1} + g_t \Delta f + \frac{1}{2} h_t \Delta f^2$ 전개, leaf 값 최적해 $w^* = -G/(H+\lambda)$ 유도, **tree 버전 Newton-Raphson 한 스텝**임을 증명, regularization $\gamma\|T\| + \frac{1}{2}\lambda \sum w_j^2$ 해석 |
| [05. LightGBM과 Histogram-based Splitting](./ch5-boosting/05-lightgbm-histogram.md) | **Gradient-based One-Side Sampling (GOSS)**의 분산 분석, **Exclusive Feature Bundling (EFB)**의 그래프 컬러링 환원, leaf-wise vs level-wise 트리 성장의 trade-off |
| [06. Boosting의 과적합 저항성](./ch5-boosting/06-boosting-margin.md) | **훈련오차 0 이후에도 테스트 오차 감소** 현상의 경험적 관찰, **Margin theory** (Schapire et al. 1998)로 설명, $P(\text{margin} \leq \theta)$의 boost 후 감소가 일반화 경계에 미치는 영향 |

</details>

<br/>

### 🔹 Chapter 6: 나이브 베이즈와 판별분석

> **핵심 질문:** Naive Bayes의 "조건부 독립" 가정이 깨져도 왜 잘 동작하는가? GNB·LDA·QDA의 결정경계는 왜 각각 선형/이차인가? LDA의 Fisher discriminant 해석은 무엇이고 PCA와 어떻게 다른가? Generative와 Discriminative는 언제 누가 이기는가?

<details>
<summary><b>Naive Bayes부터 Generative vs Discriminative까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Naive Bayes의 조건부 독립 가정](./ch6-nb-discriminant/01-naive-bayes.md) | $p(x \| y) = \prod_j p(x_j \| y)$의 엄밀한 의미, "**나이브한 가정이 깨져도 분류 경계만 정확하면 된다**" (Domingos & Pazzani 1997)의 분석, MAP 분류기 해석, Multinomial NB·Bernoulli NB·Gaussian NB의 사용 시점 |
| [02. Gaussian NB vs LDA vs QDA](./ch6-nb-discriminant/02-gnb-lda-qda.md) | GNB(대각 공분산)·LDA(공유 공분산)·QDA(클래스별 공분산)의 가정 차이, **결정경계가 선형/이차임을 직접 유도**, NumPy로 세 모델 바닥 구현 후 sklearn과 일치 검증 |
| [03. LDA의 Fisher Discriminant 해석](./ch6-nb-discriminant/03-lda-fisher.md) | between-class / within-class variance 비 $J(w) = \frac{w^\top S_B w}{w^\top S_W w}$ 최대화, **일반화 고유값 문제** $S_B w = \lambda S_W w$의 풀이, **PCA와의 차이**(분산 최대 vs 분리 최대) 시각화 |
| [04. Generative vs Discriminative](./ch6-nb-discriminant/04-generative-vs-discriminative.md) | **Naive Bayes vs Logistic Regression의 점근 비교** (Ng & Jordan 2001), 작은 $n$에서 NB가 우세, $n \to \infty$에서 LR이 우세, 데이터 크기별 **교차점 학습 곡선** |

</details>

<br/>

### 🔹 Chapter 7: KNN·클러스터링·차원축소

> **핵심 질문:** 1-NN의 점근 오차는 왜 Bayes error의 최대 2배인가(Cover-Hart)? 차원의 저주는 왜 거리 집중을 일으키는가? K-Means가 왜 GMM의 hard-assignment 극한인가? PCA·t-SNE·UMAP은 무엇을 최소화하는가?

<details>
<summary><b>KNN부터 차원축소 비교까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. K-Nearest Neighbors의 점근적 성질](./ch7-knn-clustering/01-knn-cover-hart.md) | **Cover-Hart 정리**: $n \to \infty$에서 1-NN의 오차 $\leq 2 \cdot \text{Bayes error}$, **차원의 저주** — 고차원에서 거리 집중 (concentration of distances) $\frac{\max d - \min d}{\min d} \to 0$ 증명 |
| [02. K-Means와 EM의 관계](./ch7-knn-clustering/02-kmeans-em.md) | K-Means를 **GMM의 hard-assignment 극한**으로 유도($\Sigma \to 0$), Lloyd 알고리즘의 단조감소성과 유한단계 수렴, **K-Means++**의 $O(\log k)$-경쟁비 (Arthur & Vassilvitskii 2007) |
| [03. Hierarchical Clustering](./ch7-knn-clustering/03-hierarchical.md) | Agglomerative(bottom-up) vs Divisive(top-down), linkage 기준 (single·complete·average·**Ward**)의 수학적 차이, **ultrametric**과 덴드로그램의 일대일 대응 |
| [04. DBSCAN과 밀도 기반 클러스터링](./ch7-knn-clustering/04-dbscan.md) | 핵심점·경계점·노이즈의 엄밀한 정의, **임의 모양 클러스터 탐지**의 메커니즘, $\epsilon$과 MinPts의 해석과 자동 선택, OPTICS 일반화 |
| [05. PCA·t-SNE·UMAP 비교](./ch7-knn-clustering/05-pca-tsne-umap.md) | **PCA**(선형, 분산 최대화)의 SVD 유도, **t-SNE**의 KL$(p\|\|q)$ 최소화 (van der Maaten & Hinton 2008), **UMAP**의 topological 가정 (McInnes et al. 2018), 세 방법의 local vs global 구조 보존 비교 |

</details>

---

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명**을 제공하는 대표 정리 모음입니다. 각 챕터의 문서에서 $\square$로 종결되는 엄밀한 증명을 확인할 수 있습니다. (전체 210+ 정리 중 핵심만 발췌)

| 정리 | 서술 | 출처 문서 |
|------|------|----------|
| **Normal Equation (MLE 유도)** | $y = X\beta + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2 I) \Rightarrow \hat{\beta}_{\text{MLE}} = (X^\top X)^{-1} X^\top y$ | [Ch1-01](./ch1-linear-regression/01-mle-derivation.md) |
| **잔차의 수직성** | $\hat{y} = Hy$, $H = X(X^\top X)^{-1}X^\top$ — $H^2 = H$, $r = (I-H)y \perp \text{col}(X)$ | [Ch1-02](./ch1-linear-regression/02-geometric-projection.md) |
| **Pseudoinverse의 SVD 표현** | $X = U\Sigma V^\top \Rightarrow X^+ = V\Sigma^+ U^\top$, min-norm least-squares 해의 유일성 | [Ch1-03](./ch1-linear-regression/03-pseudoinverse.md) |
| **Ridge의 3-way 등치** | $(X^\top X + \lambda I)^{-1} X^\top y$ = MAP$(\beta \sim \mathcal{N}(0, \tau^2 I))$ = $\arg\min \|y - X\beta\|^2$ s.t. $\|\beta\| \leq c$ | [Ch1-04](./ch1-linear-regression/04-ridge-three-views.md) |
| **Lasso Soft-thresholding** | Coordinate Descent 업데이트 $\beta_j \leftarrow S_\lambda(\rho_j) = \text{sgn}(\rho_j)\max(\|\rho_j\|-\lambda, 0)$ | [Ch1-05](./ch1-linear-regression/05-lasso-sparsity.md) |
| **Bias-Variance Decomposition** | $\mathbb{E}[(y-\hat{f}(x))^2] = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2$ | [Ch1-06](./ch1-linear-regression/06-bias-variance.md) |
| **Logistic Log-likelihood Concavity** | $\nabla^2 \ell(\beta) = -X^\top W X \preceq 0$ where $W = \text{diag}(p_i(1-p_i))$ — MLE 유일성 보장 | [Ch2-01](./ch2-logistic-glm/01-logistic-mle.md) |
| **Newton = IRLS** | Newton-Raphson 업데이트 $\beta^{(t+1)} = (X^\top W X)^{-1} X^\top W z$ — 가중 최소제곱 | [Ch2-02](./ch2-logistic-glm/02-irls.md) |
| **Canonical Link ⇒ Fisher Scoring = IRLS** | Exponential Family에서 canonical link 사용 시 score equation의 단순화 | [Ch2-03](./ch2-logistic-glm/03-exp-family-canonical-link.md) |
| **Information Gain = 상호정보량** | $IG(S, A) = H(S) - \sum_v \frac{\|S_v\|}{\|S\|}H(S_v) = I(Y; A)$ | [Ch3-01](./ch3-decision-tree/01-information-gain.md) |
| **Gini ≈ Entropy의 1차 Taylor** | Gini와 Entropy가 거의 같은 분할을 내림을 Taylor 전개로 증명 | [Ch3-02](./ch3-decision-tree/02-gini-vs-entropy.md) |
| **Bootstrap 포함률** | $\Pr[\text{sample } i \in \text{bootstrap}] = 1 - (1 - 1/n)^n \to 1 - 1/e$ | [Ch4-01](./ch4-bagging-rf/01-bootstrap-oob.md) |
| **Bagging Variance Formula** | $\text{Var}(\bar{f}_B) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$ — $B \to \infty$ 하한 = $\rho \sigma^2$ | [Ch4-02](./ch4-bagging-rf/02-bagging-variance-reduction.md) |
| **RF 수렴 정리 (Breiman 2001)** | $B \to \infty$에서 RF predictor가 무한 앙상블로 a.s. 수렴, generalization error 단조감소 | [Ch4-04](./ch4-bagging-rf/04-rf-convergence.md) |
| **AdaBoost 가중치 공식** | 지수손실의 FSAM ⇒ $\alpha_t = \frac{1}{2}\log\frac{1-\epsilon_t}{\epsilon_t}$, $w_i \leftarrow w_i e^{\alpha_t \mathbb{1}[y_i \neq h_t(x_i)]}$ | [Ch5-01](./ch5-boosting/01-adaboost-derivation.md) |
| **AdaBoost 훈련오차 경계** | $\frac{1}{n}\sum \mathbb{1}[H(x_i) \neq y_i] \leq \prod_t 2\sqrt{\epsilon_t(1-\epsilon_t)}$ — 지수적 감소 | [Ch5-02](./ch5-boosting/02-adaboost-theory.md) |
| **AdaBoost = GBM(지수손실)** | Gradient Boosting의 손실 함수를 $L(y, f) = e^{-yf}$로 특수화하면 AdaBoost와 동치 | [Ch5-03](./ch5-boosting/03-gradient-boosting.md) |
| **XGBoost Leaf 최적해** | $w_j^* = -G_j / (H_j + \lambda)$, gain $= \frac{1}{2}\left(\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right) - \gamma$ | [Ch5-04](./ch5-boosting/04-xgboost-second-order.md) |
| **LDA 결정경계의 선형성** | 클래스 공분산 공유 가정 하에서 log-ratio가 $w^\top x + b$ 형태 — 선형 분류기 도출 | [Ch6-02](./ch6-nb-discriminant/02-gnb-lda-qda.md) |
| **Fisher Discriminant** | $S_B w = \lambda S_W w$의 일반화 고유값 문제 — between/within 분산 비 최대화 | [Ch6-03](./ch6-nb-discriminant/03-lda-fisher.md) |
| **NB vs LR 점근 (Ng & Jordan)** | 작은 $n$에서 NB가 더 빨리 수렴, 큰 $n$에서 LR의 asymptotic error가 더 낮음 | [Ch6-04](./ch6-nb-discriminant/04-generative-vs-discriminative.md) |
| **Cover-Hart 정리** | $\lim_{n \to \infty} R_{\text{1-NN}} \leq 2 \cdot R^* (1 - R^*) \leq 2 R^*$ — 1-NN 점근 오차 경계 | [Ch7-01](./ch7-knn-clustering/01-knn-cover-hart.md) |
| **K-Means = hard-EM** | GMM의 $\Sigma \to 0$ 극한에서 K-Means의 assignment·update step 동치 | [Ch7-02](./ch7-knn-clustering/02-kmeans-em.md) |
| **K-Means++ 경쟁비** | $\mathbb{E}[\phi] \leq 8(\ln k + 2) \cdot \phi^*$ — $O(\log k)$ 근사 보장 (Arthur & Vassilvitskii 2007) | [Ch7-02](./ch7-knn-clustering/02-kmeans-em.md) |
| **PCA = SVD의 분산 최대화** | $\arg\max_{\|w\|=1} w^\top \Sigma w$ = top eigenvector — Rayleigh quotient | [Ch7-05](./ch7-knn-clustering/05-pca-tsne-umap.md) |

> 💡 **챕터별 총 정리 수**: Ch1(34) · Ch2(28) · Ch3(26) · Ch4(31) · Ch5(38) · Ch6(24) · Ch7(29) — 합계 **210개 정리 + 증명**, 약 **15,000+ 라인** 분량.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
scikit-learn==1.3.0    # 검증용 (sklearn 결과와 값 단위 비교)
xgboost==2.0.0         # Boosting Ch5 비교
lightgbm==4.1.0        # Histogram-based boosting 비교
statsmodels==0.14.0    # GLM·IRLS 비교
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            scikit-learn==1.3.0 xgboost==2.0.0 lightgbm==4.1.0 \
            statsmodels==0.14.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 — Normal Equation 3가지 방법 + 수직투영 검증
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))
beta_true = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
y = X @ beta_true + 0.1 * rng.standard_normal(100)

# ─────────────────────────────────────────────
# 방식 1: Normal Equation (직접)
# ─────────────────────────────────────────────
beta_ne = np.linalg.solve(X.T @ X, X.T @ y)

# ─────────────────────────────────────────────
# 방식 2: QR 분해 (수치적으로 안정)
# ─────────────────────────────────────────────
Q, R = np.linalg.qr(X)
beta_qr = np.linalg.solve(R, Q.T @ y)

# ─────────────────────────────────────────────
# 방식 3: SVD + Pseudoinverse (rank-deficient에도 안전)
# ─────────────────────────────────────────────
U, s, Vt = np.linalg.svd(X, full_matrices=False)
beta_svd = Vt.T @ ((U.T @ y) / s)

# sklearn과 비교
lr = LinearRegression(fit_intercept=False).fit(X, y)

print(f'Normal Equation : {beta_ne}')
print(f'QR Decomposition: {beta_qr}')
print(f'SVD Pseudoinv.  : {beta_svd}')
print(f'sklearn         : {lr.coef_}')
print(f'True beta       : {beta_true}')

# 세 방법 간 일치 (~1e-13)
print(f'\n||NE - QR||  = {np.linalg.norm(beta_ne - beta_qr):.2e}')
print(f'||NE - SVD|| = {np.linalg.norm(beta_ne - beta_svd):.2e}')

# 기하학적 검증: 잔차가 col(X)에 수직 (Ch1-02 핵심 정리)
residual = y - X @ beta_ne
print(f'\nX^T @ residual (≈ 0이어야 함): {np.max(np.abs(X.T @ residual)):.2e}')

# Hat matrix의 idempotency H^2 = H
H = X @ np.linalg.solve(X.T @ X, X.T)
print(f'||H^2 - H||_F = {np.linalg.norm(H @ H - H):.2e}')
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 방법이 ML에서 중요한가** | sklearn 한 줄과 그 뒤 수학의 간극 |
| 3 | 📐 **수학적 선행 조건** | LA·Stats·Calculus·Convex Opt·Info Theory 레포의 어떤 정리를 전제로 하는지 |
| 4 | 📖 **직관적 이해** | **확률·기하·최적화 세 가지 비유** 동시 제시 |
| 5 | ✏️ **엄밀한 정의** | Normal Equation·log-odds·Information Gain·지수손실의 엄밀한 수식 |
| 6 | 🔬 **정리와 증명** | Gauss-Markov·IRLS=Newton·Cover-Hart·AdaBoost FSAM — "자명하다" 없이 |
| 7 | 💻 **NumPy 구현 검증** | sklearn 없이 바닥부터, 그 다음 sklearn / XGBoost / LightGBM과 값 단위 비교 |
| 8 | 🔗 **실전 활용** | 언제 이 방법을 선택하는가, hyperparameter의 통계적 의미 |
| 9 | ⚖️ **가정과 한계** | 선형성·IID·Gaussian noise·조건부 독립이 깨지면? |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 — boxed equation + 표 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산 / 증명 재구성 / 구현 문제 |

> 📚 **연습문제 총 108개**: 36문서 × 문서당 3문제(기초/심화/ML 연결), 모든 문제에 `<details>` 펼침 해설 포함. Normal Equation 재유도부터 NB vs LR 점근 비교 실험까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 자동으로 다음 챕터 첫 문서로 연결되어 순차 학습이 끊기지 않습니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 410줄(증명·코드·연습문제 포함) 기준 **약 1~1.5시간**. 전체 36문서는 약 **45~55시간** 상당.

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "선형 회귀를 쓰지만 왜 그 식인지 모른다" — 회귀 집중 (5일, 약 10~13시간)</b></summary>

<br/>

```
Day 1  Ch1-01     MLE 관점에서의 Normal Equation 유도
Day 2  Ch1-02     기하학적 관점 — 수직투영과 Hat matrix
Day 3  Ch1-03     Pseudoinverse — rank-deficient 처리
Day 4  Ch1-04     Ridge의 3가지 해석 (정규화 = MAP = 제약)
Day 5  Ch1-05~06  Lasso의 sparsity와 Bias-Variance 분해
```

</details>

<details>
<summary><b>🟡 "트리 앙상블을 쓰지만 왜 작동하는지 모른다" — 트리·앙상블 집중 (1주, 약 14~17시간)</b></summary>

<br/>

```
Day 1  Ch3-01~02  Information Gain ↔ 상호정보량, Gini vs Entropy
Day 2  Ch3-03~05  회귀 트리, 가지치기, 트리의 한계 (앙상블의 동기)
Day 3  Ch4-01~02  Bootstrap과 OOB, Bagging의 분산 감소 공식
Day 4  Ch4-03~05  Random Forest와 feature subsampling, importance
Day 5  Ch5-01~02  AdaBoost의 FSAM 유도와 훈련오차 경계
Day 6  Ch5-03~04  Gradient Boosting과 XGBoost의 2차 Taylor
Day 7  Ch5-05~06  LightGBM의 GOSS·EFB, Margin theory로 본 과적합 저항성
```

</details>

<details>
<summary><b>🔴 "고전 ML의 수학적 기반을 완전 정복한다" — 전체 정복 (8주, 약 45~55시간)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — 선형 회귀의 3가지 관점
        → MLE·기하·Pseudoinverse를 한 번에 익히고
        → Ridge/Lasso의 정규화·Bayesian·제약 통합 관점

2주차  Chapter 2 전체 — Logistic Regression과 GLM
        → log-likelihood concavity, IRLS = Newton 동치
        → Exponential Family의 canonical link로 GLM 통일

3주차  Chapter 3 전체 — 결정트리
        → Information Gain ↔ 상호정보량 동치
        → CART의 Gini·MSE 분할, Cost-Complexity Pruning

4주차  Chapter 4 전체 — Bagging과 Random Forest
        → Bootstrap의 1-1/e 포함률, 분산 감소 공식 유도
        → RF의 수렴 정리 (Breiman 2001)와 feature importance

5주차  Chapter 5 전체 — Boosting (가장 분량 많음)
        → AdaBoost의 FSAM 한 줄씩 유도
        → Gradient Boosting의 함수공간 경사하강법
        → XGBoost·LightGBM의 2차 Taylor와 시스템 최적화

6주차  Chapter 6 전체 — Naive Bayes와 판별분석
        → "나이브" 가정이 왜 잘 동작하는가
        → GNB·LDA·QDA의 결정경계 유도
        → NB vs LR의 점근 비교 실험

7주차  Chapter 7 (전반) — KNN과 클러스터링
        → Cover-Hart 정리 증명 재구성
        → K-Means가 GMM의 hard-EM 극한
        → DBSCAN의 임의 모양 클러스터

8주차  Chapter 7 (후반) — 차원축소
        → PCA의 SVD 유도
        → t-SNE·UMAP의 KL/topological 목적함수 비교
        → 전체 복습: ML 4대 paradigm 지도 그리기
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | SVD, Pseudoinverse, 양정치 행렬, 수직투영 | Ch1 전체 (Normal Equation 3가지 관점), Ch6-03 (LDA의 일반화 고유값) |
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 결합/조건부 분포, 다변수정규, 분산 분해 | Ch1-06 (Bias-Variance), Ch4-02 (Bagging variance), Ch6 (NB·LDA) |
| [mathematical-statistics-deep-dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive) | MLE, Fisher 정보, Exponential Family, MAP | Ch1-01 (MLE), Ch1-04 (Ridge MAP), Ch2 전체 (GLM의 Exp Family) |
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | Gradient Descent, Newton, Coordinate Descent | Ch1-05 (Lasso CD), Ch2-02 (Newton/IRLS), Ch5-03 (함수공간 GD) |
| [convex-optimization-deep-dive](https://github.com/iq-ai-lab/convex-optimization-deep-dive) | Lagrangian, KKT, Subdifferential | Ch1-04~05 (Ridge·Lasso 쌍대), Ch5 (Boosting의 볼록 surrogate loss) |
| [information-theory-deep-dive](https://github.com/iq-ai-lab/information-theory-deep-dive) | 엔트로피, 상호정보량, KL divergence | Ch3-01 (IG = MI), Ch7-05 (t-SNE의 KL) |
| [kernel-methods-deep-dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive) | RKHS, SVM, GP, MMD, Random Features | Ch3 (트리는 비-kernel non-parametric), 비교 관점 |
| [nn-theory-deep-dive](https://github.com/iq-ai-lab/nn-theory-deep-dive) | MLP, Backprop, Universal Approximation | **이 레포의 후속** — 선형 회귀 → 1-layer NN, Logistic → softmax NN |
| [generalization-theory-deep-dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive) | VC 차원, Rademacher, NTK, 이중 강하 | Ch1-06 (Bias-Variance), Ch5-06 (Margin theory) — **이 레포의 후속** |

> 💡 이 레포는 **"sklearn 한 줄 뒤의 수학"**에 집중합니다. Linear Algebra의 Pseudoinverse를 선행하면 Ch1이, Mathematical Statistics의 Exponential Family를 선행하면 Ch2가, Information Theory의 상호정보량을 선행하면 Ch3가 훨씬 자연스럽습니다. 이 레포를 마치면 Layer 2의 [NN Theory Deep Dive](https://github.com/iq-ai-lab/nn-theory-deep-dive)와 [Generalization Theory Deep Dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive)로 자연스럽게 이어집니다.

---

## 📖 Reference

### 🏛️ 통계학습 표준 교재
- **The Elements of Statistical Learning** (Hastie, Tibshirani, Friedman, 2009) — **ESL, 표준 바이블**
- **Pattern Recognition and Machine Learning** (Bishop, 2006) — 베이지안 관점 표준
- **Machine Learning: A Probabilistic Perspective** (Murphy, 2012) — 확률적 관점 현대 표준
- **An Introduction to Statistical Learning** (James, Witten, Hastie, Tibshirani, 2013) — 쉬운 입문 (ISL)
- **The Nature of Statistical Learning Theory** (Vapnik, 1995) — SLT의 창시자

### 📐 선형 모델 · GLM
- **Generalized Linear Models** (McCullagh & Nelder, 1989) — **GLM 표준 텍스트**
- **Regression Shrinkage and Selection via the Lasso** (Tibshirani, 1996) — **Lasso 원전**
- **Ridge Regression: Biased Estimation for Nonorthogonal Problems** (Hoerl & Kennard, 1970) — Ridge 원전
- **Pathwise Coordinate Optimization** (Friedman, Hastie, Höfling, Tibshirani, 2007) — Lasso의 CD 알고리즘

### 🌳 트리·앙상블
- **Classification and Regression Trees** (Breiman, Friedman, Olshen, Stone, 1984) — **CART 원전**
- **Bagging Predictors** (Breiman, 1996) — Bagging 원전
- **Random Forests** (Breiman, 2001) — **RF 원전**
- **A Short Introduction to Boosting** (Freund & Schapire, 1999) — **AdaBoost 원전**
- **Additive Logistic Regression: A Statistical View of Boosting** (Friedman, Hastie, Tibshirani, 2000) — Boosting의 통계학적 재해석
- **Greedy Function Approximation: A Gradient Boosting Machine** (Friedman, 2001) — **Gradient Boosting 원전**
- **XGBoost: A Scalable Tree Boosting System** (Chen & Guestrin, 2016) — **XGBoost 원전**
- **LightGBM: A Highly Efficient Gradient Boosting Decision Tree** (Ke et al., 2017)
- **Boosting the Margin** (Schapire, Freund, Bartlett, Lee, 1998) — Margin theory

### 🎲 생성모델·KNN·클러스터링
- **On Discriminative vs Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes** (Ng & Jordan, 2001)
- **On the Optimality of the Simple Bayesian Classifier under Zero-One Loss** (Domingos & Pazzani, 1997) — NB가 잘 되는 이유
- **Nearest Neighbor Pattern Classification** (Cover & Hart, 1967) — **Cover-Hart 원전**
- **k-means++: The Advantages of Careful Seeding** (Arthur & Vassilvitskii, 2007) — K-Means++ 경쟁비
- **A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise** (Ester et al., 1996) — **DBSCAN 원전**
- **Visualizing Data using t-SNE** (van der Maaten & Hinton, 2008) — **t-SNE 원전**
- **UMAP: Uniform Manifold Approximation and Projection** (McInnes, Healy, Melville, 2018) — UMAP 원전

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"sklearn을 호출하는 것과, 왜 한 줄의 `.fit(X, y)` 뒤에 MLE·수직투영·Pseudoinverse·Bayesian prior·KKT·상호정보량·함수공간 경사하강법·Cover-Hart 정리가 모두 숨어 있는지를 증명할 수 있는 것은 다르다"*

</div>
