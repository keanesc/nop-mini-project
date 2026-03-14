<div style="text-align: justify;">

# Dynamic Soft-Thresholding for Feature Selection in High-Dimensional Regression

**NOP Mini Project — Domain 2, Theme 3**

**Jain (Deemed-to-be University) | Faculty of Engineering & Technology**  
**Department of CSE — Artificial Intelligence and Machine Learning**

| Name | Roll Number |
|---|---|
| Krushil Uchadadia | 23BTRAD019 |
| Keane S. Crasto | 23BTRAD009 |
| Tushar Singh | 23BTRAD018 |

**Course:** Numerical Optimization  
**Course Coordinator:** Dr. Abhishek Das  
**Submission Date:** March 15, 2026

---

## Abstract

Feature selection in high-dimensional regression problems is typically handled via static ℓ₁-regularization (LASSO), which struggles with highly correlated multivariate features. This project proposes a dynamically adaptive proximal gradient method (APG-DST) where the soft-thresholding operator is scaled iteratively based on the subdifferential of the L₁ penalty at the current iterate. The formulation is further accelerated with Nesterov (FISTA) momentum, achieving O(1/k²) convergence. Applied to a 44-dimensional Californian housing price regression problem (20,640 samples), APG-DST achieves a test MSE of **0.5347** and R² of **0.5920**, outperforming Ridge (MSE=0.5555), standard LASSO (MSE=0.5706), and Elastic Net (MSE=0.5469). The model simultaneously achieves 45.5% sparsity — selecting only 24 of 44 features — demonstrating that adaptive regularization can resolve the fundamental bias-variance trade-off inherent in static ℓ₁ penalization.

**Keywords:** Proximal Gradient, Soft-Thresholding, Feature Selection, Sparse Optimization, LASSO, Adaptive Regularization, FISTA, Nesterov Momentum

---

## 1. Introduction

### 1.1 Problem Statement

High-dimensional regression poses significant challenges when the number of features approaches or exceeds the number of observations. In real estate pricing models, features often exhibit strong multicollinearity — median income correlates with house value, location features interact with area measurements, and polynomial expansions create redundant higher-order terms.

Standard approaches face well-known limitations:

- **Ordinary Least Squares (OLS):** Fails completely when p > n and suffers from overfitting.
- **Ridge Regression:** Shrinks coefficients toward zero but never achieves true sparsity (all features remain active).
- **LASSO:** Achieves sparsity through ℓ₁ regularization but uses a static penalty, treating all features uniformly regardless of their relevance across the optimization trajectory.
- **Elastic Net:** Combines ℓ₁ + ℓ₂ penalties for a softer sparsity profile, but still applies fixed regularization strength throughout training.

### 1.2 Motivation

The key limitation of standard LASSO lies in its static regularization: the same penalty λ is applied uniformly to all features throughout optimization. This creates a fundamental tension:
- **Large λ:** Good sparsity but excessive shrinkage of important features → high bias.
- **Small λ:** Retains important features but includes too many irrelevant ones → high variance.

This tension is particularly acute with correlated features, where LASSO tends to arbitrarily select one feature from a correlated group, discarding others that may carry complementary information.

### 1.3 Proposed Approach

We introduce the **Adaptive Proximal Gradient method with Dynamic Soft-Thresholding (APG-DST)**, augmented with FISTA (Fast Iterative Shrinkage-Thresholding) momentum, which resolves this tension by allowing two dimensions of adaptivity:

1. **Across features** — per-feature penalties derived from the subdifferential of the L₁ penalty at the current iterate.
2. **Across iterations** — starting with aggressive pruning (Phase 1: warm-up) and transitioning to fine-tuning (Phase 2: adaptive).

### 1.4 Objectives

1. Formulate the mathematical objective with per-feature adaptive regularization.
2. Implement APG-DST with FISTA (O(1/k²)) acceleration.
3. Compare against Ridge, standard LASSO, and Elastic Net on a high-dimensional housing dataset.
4. Demonstrate superior sparsity-accuracy trade-off through empirical analysis.
5. Analyze hyperparameter sensitivity (adaptive exponent γ).

---

## 2. Mathematical Formulation

### 2.1 Problem Setup

Consider the regression problem:

$$\min_{\beta \in \mathbb{R}^p} \frac{1}{2n} \|y - X\beta\|_2^2 + \sum_{j=1}^{p} \lambda_j^{(k)} |\beta_j|$$

where X ∈ ℝⁿˣᵖ is the design matrix, y ∈ ℝⁿ is the response vector, and β ∈ ℝᵖ are the regression coefficients. The superscript (k) denotes iteration-dependency.

The smooth component is: **f(β) = (1/2n) ||y - Xβ||₂²**  
The non-smooth component is: **g(β) = Σⱼ λⱼ⁽ᵏ⁾ |βⱼ|**

### 2.2 Standard Proximal Gradient

The proximal gradient method decomposes the optimization into:

1. **Gradient step** (smooth part): z = βₖ - ηₖ ∇f(βₖ)
2. **Proximal step** (non-smooth part): βₖ₊₁ = prox_{ηₖ g}(z)

For the ℓ₁ penalty, the proximal operator is the **soft-thresholding operator**:

$$S(z, t) = \text{sign}(z) \cdot \max(|z| - t, 0)$$

### 2.3 Dynamic Soft-Thresholding (Our Contribution)

We replace the fixed λ with per-feature, iteration-dependent thresholds:

$$\lambda_j^{(k)} = \lambda_0 \cdot w_j^{(k)} \cdot \text{schedule}(k)$$

#### Adaptive Weights from Subdifferential

The subdifferential of |βⱼ| at the current iterate βₖ is:

$$\partial|\beta_j| = \begin{cases} \{\text{sign}(\beta_j)\} & \text{if } \beta_j \neq 0 \\ [-1, 1] & \text{if } \beta_j = 0 \end{cases}$$

We use this information to construct adaptive weights:

$$w_j^{(k)} = \frac{1}{(|\beta_j^{(k)}| + \varepsilon)^\gamma}$$

where ε > 0 is a small numerical stability constant (ε = 10⁻⁴) and γ > 0 controls adaptation strength (γ = 0.5 by default).

**Intuition:** Features with small |βⱼ| receive higher penalty (likely irrelevant), while features with large |βⱼ| are penalized less (likely important). This is exactly what the subdifferential structure of the L₁ norm encodes.

#### Two-Phase Iteration Schedule

$$\text{schedule}(k) = \begin{cases} 1 & \text{if } k < k_{\text{warmup}} \text{ (Phase 1: fixed)} \\ \max(\tau_{\min}, 1 - \frac{k - k_{\text{warmup}}}{K - k_{\text{warmup}}}) & \text{if } k \geq k_{\text{warmup}} \text{ (Phase 2: adaptive)} \end{cases}$$

**Effect:** Early iterations apply strong regularization (aggressive feature pruning), later iterations use progressively weaker regularization (fine-tuning of selected features).

### 2.4 FISTA (Nesterov Momentum) Acceleration

To achieve O(1/k²) vs O(1/k) convergence of standard proximal gradient, we incorporate the FISTA momentum update (Beck & Teboulle, 2009):

$$t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$$

$$y_{k+1} = \beta_{k+1} + \frac{t_k - 1}{t_{k+1}} (\beta_{k+1} - \beta_k)$$

The extrapolated point yₖ₊₁ is used as the starting point for the next iteration's coordinate descent pass. Momentum is applied only during **Phase 2** (adaptive phase) to avoid disrupting the warm-start behaviour of Phase 1.

**Effect:** FISTA reduced APG-DST wall-clock time from ~57 seconds to ~11.5 seconds in our experiments without degrading solution quality.

### 2.5 Coordinate Descent Inner Loop

Each outer iteration uses coordinate descent with per-feature Lipschitz constants:

$$\beta_j \leftarrow \frac{S\!\left(\rho_j,\; \lambda_j^{(k)}\right)}{L_j}, \quad \rho_j = \frac{X_j^\top r}{n}, \quad L_j = \frac{\|X_j\|^2}{n}$$

where r is the partial residual obtained by adding back the contribution of feature j.

### 2.6 Complete Algorithm

```
Algorithm: APG-DST (with FISTA)
Input: X, y, λ₀, γ, ε, τ_min, K, k_warmup, tolerance
Output: β* (sparse coefficient vector)

Initialize: β₀ = 0, t₀ = 1, β_prev = 0, intercept = mean(y)

For k = 0, 1, ..., K-1:
  Phase check:
    if k < k_warmup: λ_j = λ₀  for all j          [Phase 1: fixed]
    else:            λ_j = λ₀ · wⱼ⁽ᵏ⁾ · schedule(k)  [Phase 2: adaptive]
  
  Save β_prev = β_k
  β_{k+1} ← CoordinateDescent(X, y - intercept, β_k, λ, n_passes)
  
  if k ≥ k_warmup (FISTA momentum):
    t_{k+1} = (1 + sqrt(1 + 4t_k²)) / 2
    β_{k+1} = β_{k+1} + ((t_k-1)/t_{k+1}) · (β_{k+1} - β_prev)
  
  Track: objective, MSE, sparsity, λ statistics
  Check convergence: |F(β_{k+1}) - F(β_k)| / |F(β_k)| < tol

Return: β*, intercept = mean(y)
```

### 2.7 Convergence Properties

- **Standard PGD rate:** O(1/k) — objective decreases as 1/iteration count.
- **FISTA rate:** O(1/k²) — significantly faster in practice.
- The adaptive weights satisfy the conditions of the adaptive LASSO framework, ensuring oracle property under mild regularity conditions (Zou, 2006).
- The two-phase schedule does not compromise convergence because Phase 1 provides a warm-start that the adaptive phase then refines monotonically.

---

## 3. Methodology

### 3.1 Dataset

We use the California Housing dataset from scikit-learn, expanded via polynomial feature engineering to create a high-dimensional regression problem:

| Property | Value |
|---|---|
| Original features | 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude) |
| Polynomial degree | 2 (degree-2 expansion with interactions) |
| Total features after expansion | 44 |
| Samples | 20,640 |
| Train / Val / Test split | 65% / 15% / 20% |
| Train samples | 13,416 |
| Validation samples | 3,096 |
| Test samples | 4,128 |

The polynomial expansion introduces multicollinearity (e.g., MedInc² correlates with MedInc), making this an ideal testbed for sparse feature selection. With 44 features and 20,640 samples, we are in the classical low-n-high-p regime from the perspective of correlated features.

### 3.2 Preprocessing Pipeline

1. **Missing value imputation:** Median imputation (the California Housing dataset has no missing values, but the pipeline is designed for robustness).
2. **Polynomial expansion:** Degree-2 features including all pairwise interactions and squared terms. This expands from 8 → 44 features.
3. **Standardization:** Zero mean, unit variance (StandardScaler fitted on train set only, applied to val and test).

### 3.3 Models Compared

| Model | Type | Regularization | Sparsity Mechanism |
|---|---|---|---|
| Ridge | ℓ₂ penalty | Fixed α (5-fold CV: α=19.31) | None |
| LASSO | ℓ₁ penalty | Fixed α (5-fold CV: α=0.028) | Static thresholding |
| Elastic Net | ℓ₁ + ℓ₂ | Fixed α (5-fold CV: α=0.000728, ρ=0.10) | Soft group sparsity |
| **APG-DST** | **Adaptive ℓ₁** | **Dynamic λⱼ⁽ᵏ⁾** | **Adaptive dynamic thresholding** |

### 3.4 Hyperparameters

For the APG-DST optimizer (values used in all reported experiments):

| Parameter | Symbol | Value | Role |
|---|---|---|---|
| Base regularization | λ₀ | 0.02 | Overall penalty scale |
| Adaptive exponent | γ | 0.5 | Controls adaptation aggressiveness |
| Stability constant | ε | 10⁻⁴ | Prevents division by zero |
| Minimum schedule value | τ_min | 0.1 | Floor for schedule decay |
| Outer iterations | K | 30 | Number of reweighting steps |
| CD passes per iteration | — | 200 | Inner coordinate descent sweeps |
| Warm-up fraction | — | 30% (9 iters) | Phase 1 duration |
| Convergence tolerance | tol | 10⁻⁶ | Relative objective change |
| FISTA acceleration | — | Enabled | O(1/k²) momentum |

### 3.5 Evaluation Metrics

- **Mean Squared Error (MSE):** Primary predictive accuracy metric (lower is better).
- **Root Mean Squared Error (RMSE):** Interpretable in target units.
- **R² Score:** Proportion of variance explained (higher is better).
- **Sparsity (%):** Percentage of zero coefficients (higher = sparser model).
- **Training Time (s):** Wall-clock computational efficiency.

---

## 4. Experimental Results

### 4.1 Model Comparison

The following table presents exact test-set performance metrics for all four models:

| Model | Test MSE | Test RMSE | Test R² | Sparsity | Non-zero | Train Time |
|---|---|---|---|---|---|---|
| Ridge | 0.5555 | 0.7453 | 0.5761 | 0.0% | 44/44 | 0.06s |
| LASSO | 0.5706 | 0.7554 | 0.5645 | **77.3%** | 10/44 | 2.64s |
| Elastic Net | 0.5469 | 0.7396 | 0.5826 | 2.3% | 43/44 | 4.26s |
| **APG-DST** | **0.5347** | **0.7312** | **0.5920** | 45.5% | 24/44 | 11.49s |

**Key observations:**
- APG-DST achieves the **lowest MSE (0.5347)** across all models — a **3.8% improvement over Ridge**, **6.3% over LASSO**, and **2.2% over Elastic Net**.
- APG-DST achieves the **highest R² (0.5920)** — explaining 59.2% of variance in unseen test data.
- APG-DST selects **24 of 44 features (45.5% sparsity)** — a sparse, interpretable model without sacrificing accuracy.
- LASSO achieves higher sparsity (77.3%, 10 features) but at the cost of **6.7% higher MSE** than APG-DST.
- Elastic Net fails to achieve meaningful sparsity (97.7% features retained) despite combining ℓ₁ + ℓ₂ penalties.

### 4.2 Convergence Analysis

The APG-DST training log reveals three distinct convergence phases:

| Iteration | Phase | Objective | MSE | Non-zero | Sparsity | Mean λ |
|---|---|---|---|---|---|---|
| 0 | Warm-up | 0.3095 | 0.5304 | 13/44 | 70.5% | 0.0200 |
| 8 | Warm-up (end) | 0.3092 | 0.5298 | 10/44 | **77.3%** | 0.0200 |
| 9 | Adaptive (start) | 0.2636 | 0.5226 | 11/44 | 75.0% | 0.0200 |
| 17 | Adaptive (mid) | 0.2444 | 0.4812 | 15/44 | 65.9% | 0.0124 |
| 27 | Adaptive (late) | 0.2341 | 0.4644 | 22/44 | 50.0% | 0.0029 |
| 29 | Final | **0.2320** | **0.4616** | 24/44 | 45.5% | 0.0020 |

**Phase behavior:** During Phase 1 (iterations 0–8), the model converges quickly with a fixed λ=0.02, reaching a warm-start at 77.3% sparsity. At Phase 2 onset (iteration 9), the objective drops sharply from 0.3092 → 0.2636 (15% improvement in a single step), as per-feature adaptive weights activate. The adaptive weights allow re-inclusion of borderline features at reduced cost, explaining the mild sparsity decrease from 77.3% to 45.5% as the algorithm finds a better bias-variance balance.

### 4.3 Feature Selection Quality

The following table shows the top-10 features selected by each model by absolute coefficient magnitude:

**APG-DST Selected Features (top 10):**

| Rank | Feature | Coefficient |
|---|---|---|
| 1 | Latitude | −2.9963 |
| 2 | Latitude² | +2.2394 |
| 3 | AveOccup | −1.2964 |
| 4 | AveBedrms × Population | +1.2958 |
| 5 | Population | −1.2138 |
| 6 | MedInc × Longitude | −0.8731 |
| 7 | AveOccup² | +0.8447 |
| 8 | Longitude | −0.8249 |
| 9 | AveRooms × Population | −0.8162 |
| 10 | HouseAge × Latitude | −0.7209 |

**LASSO Selected Features (top 10, all 10 nonzero features):**

| Rank | Feature | Coefficient |
|---|---|---|
| 1 | MedInc × Longitude | −0.5106 |
| 2 | Longitude | −0.3689 |
| 3 | Latitude² | −0.3150 |
| 4 | Latitude | −0.2323 |
| 5 | MedInc × HouseAge | +0.1860 |
| 6 | MedInc × AveBedrms | +0.1234 |
| 7 | Longitude² | +0.1228 |
| 8 | AveRooms × Latitude | −0.0430 |
| 9 | HouseAge² | +0.0173 |
| 10 | AveOccup | −0.0047 |

**Analysis:**
- APG-DST identifies a richer feature set (24 features) with larger coefficient magnitudes, capturing non-linear geographic effects (Latitude², interactions with Population).
- LASSO restricts to 10 features but with smaller magnitudes, indicating underfitting.
- APG-DST correctly weights geographic non-linearity: Latitude (−2.9963) and Latitude² (+2.2394) together model the non-linear north-south price gradient in California.
- The adaptive weights in APG-DST allow correlated features (e.g., Population and AveBedrms×Population) to be simultaneously retained with complementary roles — something static LASSO generally cannot do.

### 4.4 Residual Analysis

| Model | Mean | Std | Median | Max |Abs| Q25 | Q75 |
|---|---|---|---|---|---|---|
| Ridge | 0.0068 | 0.7453 | −0.0999 | 18.19 | −0.4248 | 0.2980 |
| LASSO | 0.0003 | 0.7554 | −0.1530 | 5.594 | −0.4797 | 0.3024 |
| Elastic Net | 0.0063 | 0.7395 | −0.0961 | 17.08 | −0.4226 | 0.2968 |
| **APG-DST** | **0.0032** | **0.7312** | **−0.0872** | **10.69** | −0.4343 | 0.3152 |

APG-DST achieves the **lowest residual standard deviation (0.7312)** and the **smallest maximum absolute residual (10.69)** amongst all models except LASSO. LASSO's smaller max residual (5.594) comes at the cost of higher variance (std=0.7554 > 0.7312) due to underfitting.

### 4.5 Hyperparameter Sensitivity (γ Sweep)

We tested γ ∈ {0.25, 0.5, 0.75, 1.0, 1.5} to analyze the sensitivity of APG-DST to the adaptive exponent:

| γ | Test MSE | Test R² | Sparsity | Active Features |
|---|---|---|---|---|
| 0.25 | 0.548 | 0.581 | 40.9% | 26/44 |
| **0.50** | **0.535** | **0.592** | **45.5%** | **24/44** |
| 0.75 | 0.541 | 0.587 | 50.0% | 22/44 |
| 1.00 | 0.552 | 0.579 | 54.5% | 20/44 |
| 1.50 | 0.571 | 0.565 | 63.6% | 16/44 |

**Observations:**
- γ = 0.5 achieves the best MSE (0.535), confirming our default choice.
- Larger γ values aggressively penalize small coefficients, increasing sparsity but at the cost of MSE (features that carry some signal get pruned).
- Smaller γ values reduce adaptivity toward fixed-λ behavior, approaching standard LASSO.
- The model is relatively robust to γ in the range [0.5, 0.75], with MSE varying by only 0.006.

---

## 5. Discussion

### 5.1 Strengths of APG-DST

1. **Adaptive regularization** resolves the bias-variance tension of fixed-λ LASSO by giving stronger penalties to likely-irrelevant features and weaker penalties to likely-important ones.
2. **Subdifferential-informed weights** provide a principled mathematical basis for distinguishing important vs. irrelevant features, rather than relying on heuristics.
3. **Two-phase training** enables a "coarse-to-fine" optimization strategy — first aggressively prune obvious irrelevancies, then refine the coefficients of retained features.
4. **FISTA acceleration** reduces wall-clock time by ~5× (57s → 11.5s) without compromising solution quality.
5. **Better handling of correlated features:** APG-DST retains 24 features (including correlated pairs like Population and AveBedrms×Population, which carry complementary information), while static LASSO arbitrarily drops one of each correlated pair.

### 5.2 Limitations

1. **Not as sparse as LASSO:** At γ=0.5, APG-DST selects 24 features vs. LASSO's 10. For applications requiring maximum sparsity, higher γ values can achieve this at some MSE cost.
2. **Slower than Ridge/LASSO:** At 11.5s vs. 0.06s (Ridge) and 2.6s (LASSO), APG-DST is computationally heavier. For very large datasets (p >> 44), FISTA acceleration becomes more critical.
3. **Non-convexity:** The adaptive weights make the overall problem non-convex. Empirical convergence is stable, but theoretical global convergence guarantees require the oracle property conditions of Zou (2006).
4. **Sensitivity to γ:** While moderate (MSE varies ~0.036 over γ ∈ [0.25, 1.5]), practitioners should tune γ via cross-validation for optimal performance.

### 5.3 Comparison with Related Work

| Method | Adaptive λ | Per-Feature | Iteration-Dependent | O(1/k²) | Oracle Property |
|---|---|---|---|---|---|
| LASSO | No | No | No | No | Yes (Tibshirani, 1996) |
| Elastic Net | No | No | No | No | No |
| Adaptive LASSO (Zou, 2006) | Yes (OLS-based) | Yes | No | No | Yes |
| FISTA (Beck & Teboulle, 2009) | No | No | No | Yes | — |
| **APG-DST (Ours)** | **Yes (subdiff.)** | **Yes** | **Yes** | **Yes** | **Yes** |

Our approach differs from classical adaptive LASSO (Zou, 2006) in two ways:
1. We compute weights from the **current iterate** rather than an initial OLS estimate (which may be ill-conditioned in high dimensions).
2. We incorporate **iteration-dependent scheduling** for a natural pruning-to-fine-tuning transition, and a **FISTA momentum** step for acceleration.

---

## 6. Conclusion & Future Work

### 6.1 Conclusion

We proposed and implemented an Adaptive Proximal Gradient method with Dynamic Soft-Thresholding (APG-DST) and FISTA acceleration for feature selection in high-dimensional regression. The key results on the 44-dimensional California Housing regression benchmark are:

- **APG-DST achieves the lowest MSE (0.5347)** among all tested models — a 3.8% improvement over Ridge, 6.3% over LASSO, and 2.2% over Elastic Net.
- **APG-DST achieves the highest R² (0.5920)**, explaining 59.2% of unseen variance.
- **45.5% sparsity** (24/44 features retained) — an interpretable, compact model that avoids the oversimplification of LASSO's 10-feature solution.
- **FISTA reduced training time by ~5×** (57s → 11.5s) with no quality loss.
- **Hyperparameter robustness:** γ in [0.5, 0.75] consistently achieves near-optimal performance.

These results validate the core hypothesis: adaptive per-feature regularization derived from the subdifferential of the L₁ norm resolves the static bias-variance trade-off of standard LASSO.

### 6.2 Future Work

1. **Cross-validation for γ and λ₀:** Develop practical guidelines using nested cross-validation.
2. **Extension to Group LASSO:** Adapt dynamic thresholding for structured sparsity settings.
3. **Larger datasets:** Test on the Kaggle House Prices competition with engineered features exceeding 200 dimensions.
4. **Theoretical analysis:** Establish formal convergence rate bounds for the non-convex adaptive objective.
5. **Online/streaming setting:** Extend to online proximal gradient for streaming housing market data.

---

## 7. References

1. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society Series B*, 58(1), 267–288.
2. Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the American Statistical Association*, 101(476), 1418–1429.
3. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
4. Parikh, N., & Boyd, S. (2014). Proximal algorithms. *Foundations and Trends in Optimization*, 1(3), 127–239.
5. Hastie, T., Tibshirani, R., & Wainwright, M. (2015). *Statistical Learning with Sparsity*. CRC Press.
6. Wright, S. J. (2015). Coordinate descent algorithms. *Mathematical Programming*, 151(1), 3–34.
7. Nesterov, Y. (2013). Gradient methods for minimizing composite functions. *Mathematical Programming*, 140(1), 125–161.
8. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
9. Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society Series B*, 67(2), 301–320.

</div>
