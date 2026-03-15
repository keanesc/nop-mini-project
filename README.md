# Dynamic Soft-Thresholding for Feature Selection in High-Dimensional Regression

**NOP Mini Project — Theme 3**  
**Jain (Deemed-to-be University)** | Department of CSE — AI & DE

### Team Members

- Krushil Uchadadia (23BTRAD019)
- Keane S. Crasto (23BTRAD009)
- Tushar Singh (23BTRAD018)

---

## 📖 Overview

This project addresses feature selection in high-dimensional regression using a novel **Adaptive Proximal Gradient method with Dynamic Soft-Thresholding (APG-DST)**.

While standard LASSO applies a static penalty to all features, APG-DST dynamically adjusts the ℓ₁ penalty for each feature based on the subdifferential of the L₁ norm at the current iteration. It also incorporates **FISTA (Nesterov momentum)** for accelerated $O(1/k^2)$ convergence.

We tested our model on a 44-dimensional Californian housing dataset (expanded with degree-2 polynomial features). **APG-DST outperformed Ridge, LASSO, and Elastic Net**, achieving the lowest MSE (0.5347) and highest R² (0.5920) while simultaneously delivering a highly interpretable sparse model (45.5% sparsity).

## 🚀 How to Run the Project

### 1. Install Dependencies

Ensure you have Python 3 installed, then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard (Recommended)

We built a Flask dashboard to visually explore the results, view the convergence phases, and run live interactive predictions using the trained models.

```bash
python dashboard/app.py
```

_Open [http://localhost:5000](http://localhost:5000) in your web browser._

### 3. Run the Training Pipeline

If you want to train the models from scratch, generate the 10 evaluation plots, and run the hyperparameter sensitivity check (γ sweep):

```bash
# Run the full pipeline (takes ~60 seconds)
python main.py

# Run the pipeline FASTER (skips the gamma hyperparameter sweep, takes ~20 seconds)
python main.py --skip_gamma_sweep
```

_Results, models (`.npy`), and plots will be saved to the `results/` directory._

---

## 📊 Key Results (Test Set)

| Model       | Test MSE   | R²         | Sparsity  | Training Time |
| ----------- | ---------- | ---------- | --------- | ------------- |
| Ridge       | 0.5555     | 0.5761     | 0.0%      | ~0.06s        |
| LASSO       | 0.5706     | 0.5645     | 77.3%     | ~2.64s        |
| Elastic Net | 0.5469     | 0.5826     | 2.3%      | ~4.26s        |
| **APG-DST** | **0.5347** | **0.5920** | **45.5%** | **~11.49s**   |

_APG-DST successfully balances high predictive accuracy (beating all baselines) with significant sparsity (selecting 24 out of 44 features)._

---

## 📂 Repository Structure

- `src/optimizers/adaptive_proximal.py`: Core logic for APG-DST with FISTA acceleration.
- `src/models/baselines.py`: Wrappers for Ridge, LASSO, and Elastic Net comparisons.
- `src/train.py`: Pipeline execution and hyperparameter sweep (`run_gamma_sweep`).
- `src/visualize.py`: Matplotlib/Seaborn visualization suite.
- `dashboard/app.py`: Flask application for real-time visualization and interaction.
- `report/report.md`: The complete mathematical derivation and technical report.
- `results/`: Contains the generated training statistics, saved models, and all PNG plots.
