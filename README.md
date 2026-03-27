# Credit Risk Modelling — Give Me Some Credit

End-to-end credit-default modelling pipeline on the [Kaggle *Give Me Some Credit*](https://www.kaggle.com/c/GiveMeSomeCredit) dataset. Covers exploratory data analysis, a Mixture-of-Experts neural network, gradient boosting models, and knowledge distillation — all implemented in Python with Jupyter notebooks.

---

## Dataset

| Property | Value |
| --- | --- |
| Source | Kaggle — *Give Me Some Credit* (2011) |
| Rows | ~150,000 borrowers |
| Features | 10 financial / delinquency variables |
| Target | `SeriousDlqin2yrs` — 1 if 90+ days past due within 2 years |
| Default rate | ~6.7% (class imbalance ratio ≈ 13.5 : 1) |

**Feature reference:**

| Feature | Description |
| --- | --- |
| `unsecured_credit` | Revolving utilisation of unsecured lines (0–1+) |
| `age` | Borrower age in years |
| `delinq_30_59` | Times 30–59 days past due in last 2 years |
| `debt_ratio` | Monthly debt payments / monthly gross income |
| `monthly_income` | Monthly gross income ($) |
| `open_credit` | Number of open credit lines and loans |
| `delinq_90` | Times 90+ days past due |
| `real_estate_loans` | Number of real estate loans or lines |
| `delinq_60_89` | Times 60–89 days past due in last 2 years |
| `dependents` | Number of dependents |

---

## Project Structure

```
DSL_LAB/
├── Data/
│   ├── cs-training.csv              # training set (~150k rows)
│   ├── cs-test.csv                  # test set
│   ├── sampleEntry.csv              # sample submission
│   └── Data Dictionary.xls         # official feature descriptions
├── notebooks/
│   ├── eda.ipynb                    # exploratory data analysis
│   ├── moe_model.ipynb              # Mixture-of-Experts (raw income)
│   ├── moe_model_with_log_income.ipynb  # MoE with log-transformed income
│   └── boosting_models.ipynb        # XGBoost + CatBoost + Optuna
├── utils/
│   └── download_data.py
├── SESSION_SUMMARY.md               # detailed implementation reference
├── pyproject.toml                   # uv dependencies
└── README.md
```

---

## Notebooks

### `eda.ipynb` — Exploratory Data Analysis (40 cells)

A comprehensive analysis of the raw data before any modelling.

**Sections:**
- **Missingness Analysis** — `missingno` matrix, MNAR chi-square test for `monthly_income` vs default status, binary missingness indicator creation
- **Univariate Analysis** — distributions for continuous and discrete features; log-transform demonstration for `monthly_income` (raw skewness ≈ 5–8 → near 0 after `log1p`)
- **Bivariate Analysis** — violin plots by default status, default rate by age group and delinquency count
- **Correlation Analysis** — Pearson and Spearman heatmaps, ranked target correlations, pairplot of top features
- **VIF Analysis** — Variance Inflation Factors via `statsmodels`; color-coded bar chart (blue = fine, orange = VIF > 5, red = VIF > 10)
- **Mutual Information** — `sklearn` k-NN MI estimator; side-by-side comparison with Spearman correlation
- **Baseline Logistic Regression** — standardised coefficients with odds ratios, McFadden pseudo-R², ROC-AUC, Average Precision

**Key findings:**
- `delinq_90`, `delinq_30_59`, `delinq_60_89` are the strongest predictors by both Spearman and MI
- `monthly_income` is heavily right-skewed → `log1p` recommended for linear models
- `debt_ratio` and credit-line counts show multicollinearity (VIF)

---

### `moe_model_with_log_income.ipynb` — Mixture-of-Experts Neural Network (31 cells)

A **soft Mixture-of-Experts** classifier with a gating network that learns to route each borrower to the most appropriate expert. Uses `log1p(monthly_income)` as an input feature.

**Architecture:**

```
Input (12 features)
    ├── Expert 0: Linear(12,64) → BN → ReLU → Dropout → Linear(64,64) → BN → ReLU → Dropout → Linear(64,1)
    ├── Expert 1: (same)
    ├── Expert 2: (same)
    └── GatingNet: Linear(12,32) → ReLU → Linear(32,3) → Softmax

Output logit = Σ  gate_k × expert_k_logit
```

**Training:**
- Loss: `BCEWithLogitsLoss(pos_weight=7.0)` + Switch Transformer auxiliary load-balancing loss
- Optimiser: Adam (`lr=1e-3`, `weight_decay=1e-4`) + `ReduceLROnPlateau`
- Early stopping on val AUC (patience=100, max 400 epochs)
- Two threshold strategies: max-F1 and max-recall s.t. precision ≥ 50%

**Sections:**
- **Expert Specialisation** — load distribution, gate weight distributions, default rate per expert
- **Per-Expert Deep Analysis** — gradient-based feature importance (`|∂f_k/∂x|` per expert), feature profiles (mean standardised values per expert's dominant samples), per-expert AUC / F1 / Precision / Recall with radar chart
- **UMAP Projection** — 4-panel: expert territories, actual class, predicted class, gate confidence
- **Knowledge Distillation** — 2-layer NN student (`Linear→ReLU→Dropout→Linear`) trained with temperature-scaled soft labels (T=3); comparison: MoE Teacher vs NN+Soft KD vs NN Hard Labels

---

### `moe_model.ipynb` — Mixture-of-Experts (raw income)

Same architecture as above but using raw `monthly_income` (no log transform). Kept as a baseline to isolate the effect of the income transformation.

---

### `boosting_models.ipynb` — Gradient Boosting (29 cells)

Tree-based models as a complement to the MoE neural network.

**Models:**

| Model | Key settings |
| --- | --- |
| XGBoost | `n_estimators=2000`, `lr=0.05`, `max_depth=6`, `scale_pos_weight=n_neg/n_pos`, early stopping |
| CatBoost | `iterations=2000`, `lr=0.05`, `depth=6`, ordered boosting, `l2_leaf_reg=3`, early stopping |
| CatBoost (Optuna) | TPE sampler, 50 trials, tunes `lr`, `depth`, `l2_leaf_reg`, `bagging_temperature`, `random_strength`, `border_count` |

**Sections:**
- **Shared evaluation helpers** — `get_thresholds()`, `evaluate_model()`, `plot_results()` (reused for all models)
- **XGBoost feature importance** — weight, gain, cover side-by-side
- **CatBoost feature importance** — `PredictionValuesChange` and `LossFunctionChange`
- **Optuna tuning** — optimisation history and parameter importance plots
- **Model comparison** — overlaid ROC and PR curves, normalised feature importance comparison

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone
git clone https://github.com/metaphorpritam/DSL_LAB_CRM.git
cd DSL_LAB_CRM

# Install dependencies
uv sync

# Launch Jupyter
uv run jupyter notebook
```

**Key dependencies:** `torch`, `xgboost`, `catboost`, `scikit-learn`, `umap-learn`, `optuna`, `statsmodels`, `missingno`, `matplotlib`, `seaborn`

---

## Evaluation Metrics

All models are evaluated with:

| Metric | Why |
| --- | --- |
| **ROC-AUC** | Ranking quality; threshold-independent |
| **Average Precision** | Calibrated ranking; sensitive to class imbalance |
| **F1 Score** | Balance of precision and recall at chosen threshold |
| **McFadden pseudo-R²** | Goodness-of-fit for logistic models |

Two threshold strategies are applied to every classifier:
- **Strategy A** — threshold that maximises F1 on the validation set
- **Strategy B** — highest recall threshold s.t. Default-class precision ≥ 50%

---

## References

- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features*. NeurIPS.
- Fedus, W. et al. (2022). *Switch Transformers: Scaling to Trillion Parameter Models*. JMLR.
- McInnes, L. et al. (2018). *UMAP: Uniform Manifold Approximation and Projection*. JOSS.
- Rubin, D.B. (1976). *Inference and Missing Data*. Biometrika.
