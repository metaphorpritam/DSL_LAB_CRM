# DSL_LAB — Session Summary

**Last updated:** 2026-03-28 · Give Me Some Credit (Kaggle) · Credit-Default Modelling · session 3 additions: prediction script with Google Sheet integration, CIBIL scoring, SHAP explanations

---

## 1. Project Context

**Dataset**: Kaggle *Give Me Some Credit* (2011) — `Data/cs-training.csv`

- 150,000 borrowers; 10 features; binary target `SeriousDlqin2yrs`
- **Target = 1** if the borrower had 90+ days past-due delinquency within 2 years
- Default rate ≈ 6.7% → class imbalance ratio ≈ 13.5:1
- Basel II/III context: PD estimation has direct regulatory and capital consequences

**Toolchain**: Python 3 + uv, Jupyter, PyTorch, XGBoost, CatBoost, scikit-learn, statsmodels, umap-learn, Optuna, missingno, SHAP, Plotly

---

## 2. Shared Data Pipeline

Every notebook uses this exact pipeline (copy-pasted, not imported):

```python
raw = pd.read_csv('../Data/cs-training.csv', index_col=0)

RENAME = {
    'SeriousDlqin2yrs':                        'defaulted',
    'RevolvingUtilizationOfUnsecuredLines':     'unsecured_credit',
    'age':                                      'age',
    'NumberOfTime30-59DaysPastDueNotWorse':     'delinq_30_59',
    'DebtRatio':                                'debt_ratio',
    'MonthlyIncome':                            'monthly_income',
    'NumberOfOpenCreditLinesAndLoans':          'open_credit',
    'NumberOfTimes90DaysLate':                  'delinq_90',
    'NumberRealEstateLoansOrLines':             'real_estate_loans',
    'NumberOfTime60-89DaysPastDueNotWorse':     'delinq_60_89',
    'NumberOfDependents':                       'dependents',
}
raw = raw.rename(columns=RENAME)

df = raw.copy()
df = df[df['age'] > 0]           # removes impossible age=0 rows

# Sentinel treatment: flag codes 96/98 (data-entry codes), cap all delinquency cols at 10
DELINQ_COLS = ['delinq_30_59', 'delinq_60_89', 'delinq_90']
df['delinq_sentinel'] = (df['delinq_90'] >= 96).astype(int)   # binary flag: 1 if sentinel
for col in DELINQ_COLS:
    df[col] = df[col].clip(upper=10)                            # cap extreme values

# Missing indicators derived from raw (re-execution-safe: not affected by imputation order)
df['monthly_income_missing'] = raw.loc[df.index, 'monthly_income'].isna().astype(int)
df['dependents_missing']     = raw.loc[df.index, 'dependents'].isna().astype(int)

# Imputation
df['monthly_income'] = df['monthly_income'].fillna(df['monthly_income'].median())
df['dependents']     = df['dependents'].fillna(0)
```

**Note**: Earlier versions filtered `df = df[df['delinq_90'] < 96]` (dropping sentinels). Current pipeline retains those rows and encodes them with `delinq_sentinel=1` while capping the raw count at 10.

**Missingness facts**:

- `monthly_income`: 19.8% missing — MNAR (higher default rate when income is missing, chi-square confirmed p<0.05)
- `dependents`: 2.6% missing — imputed with 0 (low MI, conservative assumption)

**Feature set used in MoE / boosting models** (13 total):

```python
FEATURE_COLS = [
    'unsecured_credit', 'age', 'delinq_30_59', 'debt_ratio',
    'monthly_income', 'open_credit', 'delinq_90', 'real_estate_loans',
    'delinq_60_89', 'dependents', 'monthly_income_missing', 'dependents_missing',
    'delinq_sentinel',   # added: binary flag for sentinel rows (delinq_90 >= 96)
]
```

**Train/val/test split** (all notebooks, SEED=42):

```python
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp)
# Result: 70% train / 15% val / 15% test
```

---

## 3. `notebooks/eda.ipynb` (40 cells)

### 3.1 Imports

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.ticker as mticker, seaborn as sns, missingno as msno
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

SEED = 42
GOOD_CLR = '#4C9BE8'   # non-default colour
BAD_CLR  = '#E8604C'   # default colour
TARGET_PALETTE = {0: GOOD_CLR, 1: BAD_CLR}
```

### 3.2 Data Cleaning (exact filters)

```python
df = raw.copy()
df = df[df['age'] > 0]
df = df[df['delinq_90'] < 96]
# Note: unsecured_credit > 1.0 and debt_ratio > 1.0 are RETAINED as valid signals
```

### 3.3 Missingness Analysis

Three `missingno` plots: matrix (row-level patterns), bar (completeness per column), heatmap (co-missingness).

MNAR test:

```python
df['income_missing'] = df['monthly_income'].isnull().astype(int)
ct = pd.crosstab(df['income_missing'], df['defaulted'])
chi2, p_val, _, _ = stats.chi2_contingency(ct)
# → missingness IS significantly associated with default (p < 0.05)
```

### 3.4 Univariate Analysis

- Continuous features (`age`, `unsecured_credit`, `debt_ratio`, `monthly_income`): histogram + KDE, x-axis clipped to `min(p99, median + 5*IQR)`
- Discrete features: bar charts capped at 97th percentile
- Boxplots: `LOG_SCALE_COLS = {"unsecured_credit", "debt_ratio", "monthly_income"}` use log y-scale

**Log-transform cell** (inserted after monthly income plot):

```python
log_income = np.log1p(df['monthly_income'])
# Side-by-side: raw (clipped p99) vs log1p histogram
print(f"Raw skewness   : {df['monthly_income'].skew():.2f}")   # ≈ 5-8
print(f"Log1p skewness : {log_income.skew():.2f}")              # near 0
```

### 3.5 Bivariate Analysis

- Violin+box plots split by `defaulted` for 7 features
- Default rate by age group: `pd.cut(df['age'], bins=range(20, 100, 5))` — dual-axis bar+line
- Default rate by delinquency count: each delinq col capped at 10, dual-axis
- Mean feature values table: `df.groupby('defaulted')[feat_cols].mean().T` with `Ratio (Default/Non-Default)` column

### 3.6 Correlation Analysis

```python
# Pearson and Spearman heatmaps side-by-side (lower triangle only, mask upper)
corr = df.corr(method='pearson'/'spearman', numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))

# Ranked bar chart: target correlation
target_corr = df.corr(method='spearman', numeric_only=True)['defaulted'].drop('defaulted').sort_values(key=abs)
# red = positive correlation with default; blue = negative

# Pairplot: top 5 features by |correlation| + target, 3000-row sample
top_feats = target_corr.abs().nlargest(5).index.tolist() + ['defaulted']
```

### 3.7 VIF Analysis (inserted after pairplot)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

VIF_COLS = ['unsecured_credit', 'age', 'delinq_30_59', 'debt_ratio',
            'monthly_income', 'open_credit', 'delinq_90', 'real_estate_loans',
            'delinq_60_89', 'dependents']

X_vif = add_constant(df[VIF_COLS].dropna())
vif_df = pd.DataFrame({
    'Feature': VIF_COLS,
    'VIF': [variance_inflation_factor(X_vif.values, i + 1) for i in range(len(VIF_COLS))],
}).sort_values('VIF', ascending=False)
# Color coding: crimson=VIF>10 (severe), orange=VIF>5 (moderate), steelblue=fine
```

### 3.8 Mutual Information

```python
discrete = ['delinq_30_59', 'delinq_60_89', 'delinq_90', 'open_credit', 'real_estate_loans', 'dependents']
discrete_mask = [col in discrete for col in X.columns]
mi_scores = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=SEED)
# sklearn uses k-NN estimator for continuous features
```

Side-by-side comparison: Spearman `|r|` vs MI — sorted by MI descending.

### 3.9 Key Findings (Section 10)

| # | Finding | Implication |
| --- | --- | --- |
| 1 | Class imbalance ~13.5:1 | Use class weights / SMOTE / threshold tuning; accuracy is misleading |
| 2 | `monthly_income` 19.8% missing, MNAR | Median imputation for baseline; multiple imputation for production |
| 3 | `dependents` 2.6% missing, low MI | Zero imputation; could be dropped |
| 4 | `delinq_90`, `delinq_30_59`, `delinq_60_89` strongest predictors (both Spearman + MI) | Include all three; check multicollinearity |
| 5 | `unsecured_credit` moderate positive correlation | High utilisation = financial stress |
| 6 | `age` negatively correlated | Older borrowers default less; non-linear relationship |
| 7 | `debt_ratio` + `open_credit` / `real_estate_loans` correlated → VIF risk | L2 regularisation helps for LR |
| 8 | `monthly_income` heavily right-skewed | log1p before linear models |

### 3.10 Baseline Logistic Regression (Section 11)

```python
LR_COLS = ['unsecured_credit', 'age', 'delinq_30_59', 'debt_ratio',
           'monthly_income', 'open_credit', 'delinq_90', 'real_estate_loans',
           'delinq_60_89', 'dependents']

df_lr = df[LR_COLS + ['defaulted']].copy()
df_lr['monthly_income'] = np.log1p(df_lr['monthly_income'])   # log transform here

X_tr, X_te, y_tr, y_te = train_test_split(X_lr, y_lr, test_size=0.20, random_state=42, stratify=y_lr)
scaler_lr = StandardScaler()
X_tr_s = scaler_lr.fit_transform(X_tr)
X_te_s = scaler_lr.transform(X_te)

lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr.fit(X_tr_s, y_tr)

# McFadden pseudo-R²:
null_prob = y_tr.mean()
ll_null   = -log_loss(y_te, np.full_like(probs, null_prob)) * len(y_te)
ll_model  = -log_loss(y_te, probs) * len(y_te)
pseudo_r2 = 1 - ll_model / ll_null

# Outputs: intercept, coefficient table (sorted by |coef|) with odds ratios,
#          ROC-AUC, Avg Precision, McFadden R², Log-Loss, classification_report
# Plot: horizontal bar chart — crimson=positive coef, steelblue=negative coef
```

**Note**: LR uses 80/20 split (not 70/15/15) and only 10 features (no missing indicators).

---

## 4. `notebooks/moe_model_with_log_income.ipynb` (31 cells)

**Key differences from `moe_model.ipynb`**:

1. `monthly_income` log1p-transformed after imputation (reduces right skew ≈5–8 → near 0; helps early-epoch convergence; BatchNorm mitigates residual skew)
2. Sentinel handling: `delinq_sentinel` flag + `.clip(upper=10)` replacing old row-filter
3. `FEATURE_COLS` now 13 features (+ `delinq_sentinel`)
4. KD student is a 2-layer `StudentNet` NN (not LR as in original `moe_model.ipynb`)

```python
# Applied after imputation so the filled median is also consistently transformed
df['monthly_income'] = np.log1p(df['monthly_income'])
```

### 4.1 MoE Theory (exact equations from notebook)

Soft-MoE output:

```text
ŷ = Σ_{k=1}^{K}  g_k(x) · f_k(x)

where:
  g_k(x) = softmax(W_g · x + b_g)_k        ← gate weight for expert k
  f_k(x) = Expert_k MLP logit               ← 3-layer MLP with BN + Dropout
  sigmoid(ŷ) = default probability
```

Switch Transformer auxiliary loss (Fedus et al., 2022) — prevents expert collapse:

```text
L_aux = α · K · Σ_{k=1}^{K}  f_k · P_k

where:
  f_k = fraction of batch hard-routed to expert k  (non-differentiable: argmax)
  P_k = mean gate weight for expert k               (differentiable: mean of softmax)
  α = 0.01,  K = 3
  At perfect balance: f_k = P_k = 1/K → L_aux = α

Total loss: L = L_BCE(pos_weight) + L_aux
```

### 4.2 Model Classes (exact code)

```python
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x)   # shape: (B, 1)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, n_experts=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_experts),
        )
    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)   # shape: (B, K)

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, n_experts=3, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, dropout) for _ in range(n_experts)])
        self.gate = GatingNetwork(input_dim, n_experts)
    def forward(self, x):
        gate_weights = self.gate(x)                                       # (B, K)
        expert_outs  = torch.cat([e(x) for e in self.experts], dim=1)    # (B, K)
        logits = (gate_weights * expert_outs).sum(dim=1)                  # (B,)
        return logits, gate_weights

def auxiliary_loss(gate_weights, alpha=0.01):
    K = gate_weights.shape[1]
    top_k = gate_weights.argmax(dim=1)                    # hard routing, non-differentiable
    f_k = torch.zeros(K, device=gate_weights.device)
    for k in range(K):
        f_k[k] = (top_k == k).float().mean()
    P_k = gate_weights.mean(dim=0)                        # differentiable
    return alpha * K * (f_k * P_k).sum()
```

### 4.3 Hyperparameters (exact values from notebook)

```python
SEED        = 42
HIDDEN_DIM  = 64
DROPOUT     = 0.3
N_EXPERTS   = 3
BATCH_SIZE  = 512
LR          = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS  = 400
PATIENCE    = 100          # early stopping on val AUC
AUX_ALPHA   = 0.01
TEMPERATURE = 3.0          # for knowledge distillation soft labels
POS_WEIGHT  = 7.0          # manual override (data ratio ≈14); lower = more precision, less recall
```

### 4.4 Training Loop

```python
bce_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Per epoch: forward pass → BCE + aux_loss → backward → step → scheduler.step(val_auc)
# Best state saved by val AUC; restored after patience_counter >= PATIENCE
```

### 4.5 Threshold Strategies (Section 8)

```python
PREC_TARGET = 0.50   # minimum Default-class precision

# Strategy A — max F1: thresholds[argmax(f1_vals)]
# Strategy B — max recall s.t. precision >= PREC_TARGET
feasible = prec_vals >= PREC_TARGET
best_idx = feasible.nonzero()[0][np.argmax(rec_vals[feasible])]

# Active threshold used downstream: best_thresh = thresh_prec  (Strategy B)
```

### 4.6 Expert Specialisation (Section 9)

```python
dominant_expert = test_gate_w.argmax(axis=1)   # (N_test,)
gate_confidence = test_gate_w.max(axis=1)

# Per expert: dominant %, mean gate weight, default rate
# Plots: bar chart of default rate per expert, gate weight distributions, gate confidence histogram
```

### 4.7 Per-Expert Deep Analysis (Section 9b — added cell)

**Gradient-based feature importance**:

```python
model.eval()
all_grads = [[] for _ in range(N_EXPERTS)]
for i in range(0, len(X_test_s), 256):
    Xb = torch.tensor(X_test_s[i:i+256], dtype=torch.float32, requires_grad=True).to(device)
    for k in range(N_EXPERTS):
        out = model.experts[k](Xb).sum()
        grad = torch.autograd.grad(out, Xb, retain_graph=(k < N_EXPERTS - 1))[0]
        all_grads[k].append(grad.abs().detach().cpu().numpy())
# expert_importance[k] = mean |∂expert_k(x)/∂x|, normalised to sum=1 per expert
# Plots: heatmap (feature × expert) + grouped bar chart
```

**Feature profiles** (borrower archetypes per expert):

```python
profiles = np.array([X_test_s[dominant_expert == k].mean(axis=0) for k in range(N_EXPERTS)])
# Heatmap: expert × feature, values = mean standardised feature value
# Reveals which borrower segment each expert handles
```

**Per-expert metrics**:

```python
for k in range(N_EXPERTS):
    mask = dominant_expert == k
    auc_k = roc_auc_score(y_test[mask], test_probs[mask])
    f1_k  = f1_score(y_test[mask], (test_probs[mask] >= best_thresh).astype(int))
    # + precision, recall
# Radar/spider chart comparing all three experts on AUC, F1, Precision, Recall
```

### 4.8 UMAP 2D (Section 10)

```python
np.random.seed(SEED)
umap_idx  = np.random.choice(len(X_train_s), size=min(8000, len(X_train_s)), replace=False)
reducer   = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
embedding = reducer.fit_transform(X_umap)

# 4-panel 2x2 figure:
# Panel 1: dominant expert (colour by argmax gate weight)  — ec = ['#2196F3', '#FF5722', '#4CAF50']
# Panel 2: actual class (coolwarm colormap)
# Panel 3: predicted class (coolwarm colormap)
# Panel 4: gate confidence / max gate weight (viridis colormap)
```

### 4.9 UMAP 3D & Animations (Section 10b) — presentation section

Same 8 000-sample subsample re-used; UMAP refit with `n_components=3`.

```python
reducer_3d   = umap.UMAP(n_components=3, random_state=SEED, n_neighbors=30, min_dist=0.1)
embedding_3d = reducer_3d.fit_transform(X_umap)   # (8000, 3)
```

Four sub-sections:

| Sub-section | Tool | Output |
| --- | --- | --- |
| 10b.1 Static 3D scatter | `mpl_toolkits.mplot3d` | 2×2 figure matching 2D panels; fixed `elev=25, azim=45` |
| 10b.2 Rotating animation — expert territories | `FuncAnimation` → `to_jshtml()` | Single panel, 120 frames × 3°/frame, elevation bobs with sine wave |
| 10b.3 4-panel synchronised rotation | `FuncAnimation` → `to_jshtml()` | All four colourings rotate in lockstep, 90 frames × 4°/frame |
| 10b.4 Interactive Plotly 3D | `plotly.graph_objects.Scatter3d` | Drag-to-orbit; dropdown switches colouring (expert / actual / predicted / confidence) |

Animation notes:

- `matplotlib.rcParams['animation.embed_limit'] = 64` prevents truncation of jshtml output
- Data drawn once before the loop; only `ax.view_init(elev, azim)` called per frame → fast render
- `plt.close(fig)` before `HTML(anim.to_jshtml())` suppresses the duplicate static frame

### 4.10 Knowledge Distillation (Section 11)

**Soft labels**:

```python
# Temperature scaling: p_soft = σ(logit_teacher / T)
TEMPERATURE = 3.0
# Higher T flattens distribution → richer signal about relative teacher confidence
```

**Student architecture** (2-layer NN, replacing LR from original):

```python
class StudentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)   # logit output
```

**Training function**:

```python
def train_student(X_tr, y_tr_soft, X_v, y_v_hard, mode='kd',
                  epochs=100, patience=10, pw=1.0):
    # mode='kd'   → BCEWithLogitsLoss with soft targets (continuous [0,1]) as labels
    # mode='hard' → BCEWithLogitsLoss with binary targets + pos_weight
    # Optimiser: Adam lr=1e-3, weight_decay=1e-4
    # Scheduler: ReduceLROnPlateau(mode='max', factor=0.5, patience=3) on val AUC
    # Early stopping: patience=10 on val AUC; saves best state
```

**KD rationale**: `BCEWithLogitsLoss(logit, soft_target)` directly minimises KL divergence
between student and teacher distributions. No sklearn duplication trick needed (unlike
original LR student).

**Three models compared**: MoE Teacher | NN + Soft KD | NN Hard Labels

Metrics: ROC-AUC, Avg Precision, F1, Spearman rank correlation with teacher probabilities

---

## 5. `notebooks/boosting_models.ipynb` (29 cells)

### 5.1 Shared Helpers (Section 3)

```python
PREC_TARGET = 0.50   # minimum acceptable Default-class precision

def get_thresholds(y_val, val_probs):
    # Returns thresh_f1 (max F1) and thresh_prec (max recall s.t. precision >= PREC_TARGET)
    thresholds = np.linspace(0.01, 0.99, 499)
    ...

def evaluate_model(name, y_test, test_probs, thresh_f1, thresh_prec):
    # Prints classification reports for both strategies; returns results dict

def plot_results(name, y_test, test_probs, test_preds_f1, test_preds_prec, thresh_f1, thresh_prec):
    # 4-panel: confusion matrix A, confusion matrix B, ROC curve, PR curve
```

### 5.2 XGBoost (Section 4)

```python
xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,        # was 1; higher = more conservative splits
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=class_ratio,   # handles class imbalance (n_neg/n_pos)
    eval_metric='aucpr',       # was 'auc'; AUC-PR more informative under imbalance
    early_stopping_rounds=30,  # was 50
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)
xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=100)
```

**Feature importance types**:

- `weight`: number of times feature is used to split across all trees
- `gain`: average information gain per split (most meaningful)
- `cover`: average number of samples covered per split

### 5.3 CatBoost (Section 5)

```python
train_pool = Pool(X_train_s, y_train, feature_names=FEATURE_COLS)
val_pool   = Pool(X_val_s,   y_val,   feature_names=FEATURE_COLS)

cat_model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    class_weights=[1, class_ratio],   # list format [w_neg, w_pos]; was dict {0:1, 1:ratio}
    eval_metric='AUC',
    early_stopping_rounds=30,         # was 50
    random_seed=SEED,
    verbose=100,
)
cat_model.fit(train_pool, eval_set=val_pool)   # uses Pool objects (exposes feature names)
```

**CatBoost innovations**:

- **Ordered boosting**: prevents target leakage by using a permuted history per sample
- **Symmetric (oblivious) trees**: same split criterion at each level → faster inference, reduces overfitting

**Feature importance types**:

- `PredictionValuesChange`: how much model predictions change when feature is removed
- `LossFunctionChange`: how much loss changes when feature is removed (more rigorous)

### 5.4 Optuna Hyperparameter Tuning (Section 6)

```python
import optuna
from optuna.samplers import TPESampler

N_TRIALS = 50

def objective(trial):
    params = {
        'learning_rate':       trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth':               trial.suggest_int('depth', 4, 10),
        'iterations':          trial.suggest_int('iterations', 500, 3000),   # added
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength':     trial.suggest_float('random_strength', 0.0, 10.0),
        'border_count':        trial.suggest_int('border_count', 32, 255),
        # fixed: class_weights, eval_metric, early_stopping_rounds=50, random_seed
    }
    # Trains CatBoost with these params, returns val AUC

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
# Plots: optimisation history, parameter importance
```

**TPE (Tree-structured Parzen Estimator)**: models p(params | good) and p(params | bad) as kernel density estimates, samples from ratio — more efficient than random/grid search.

### 5.5 Model Comparison (Section 7)

Models compared: XGBoost | CatBoost baseline | CatBoost tuned (Optuna)

Plots:

- Overlaid ROC curves (all 3 models, both thresholds)
- Overlaid PR curves
- Normalised feature importance: XGBoost `gain` vs CatBoost `PredictionValuesChange` (side-by-side bar chart)

---

## 6. `notebooks/summary.ipynb` (46 cells)

Self-contained consolidation notebook — runs all four models end-to-end then compares them.

### 6.1 Structure

| Section | Cells | Content |
| --- | --- | --- |
| §1 Data pipeline | 2 | Same shared pipeline (sentinel, imputation, 13 features) |
| §2.1 Data profiling | 1 | `describe()` + missing % + skewness table |
| §2.2 Missingness | 1 | Completeness bar + MNAR chi-square (default rate by income presence) |
| §2.3 Distributions | 1 | Continuous histograms (raw p99 + log-scale row), discrete bar charts |
| §2.4 Correlation | 1 | Spearman heatmap + signed r vs normalised MI side-by-side |
| §3 Split | 1 | 70/15/15 stratified; shared `get_thresholds` + `model_metrics` helpers |
| §4.1 Logistic Regression | 1 | 13 features, log1p income, `class_weight='balanced'` |
| §4.2 MoE Teacher | 5 | Full model + training loop (MAX_EPOCHS=200, PATIENCE=30 for speed) |
| §4.3 KD Student | 2 | Temperature-scaled soft labels (T=3), 2-layer `StudentNet` |
| §4.4 CatBoost + Optuna | 2 | TPE search N_TRIALS=30, retrain with best params |
| §5 Comparison | 5 | Metrics table (both threshold strategies), overlaid ROC/PR, bar chart, Optuna history |
| §6 SHAP | 9 | Per-model beeswarm + mean \|SHAP\| bars + cross-model heatmap & rank table |
| §7 Summary | 1 | EDA findings table + model recommendation narrative |

### 6.2 SHAP Feature Importance (Section 6)

| Model | Explainer | Notes |
| --- | --- | --- |
| Logistic Regression | `shap.LinearExplainer` | Exact; `maskers.Independent` background |
| MoE Teacher | `shap.GradientExplainer` | `MoELogitWrapper` strips gate output before SHAP sees model |
| KD Student | `shap.GradientExplainer` | Same 300-sample background tensor |
| CatBoost (Optuna) | `shap.TreeExplainer` | Exact tree SHAP; guards against `list` return for binary |

`SHAP_N=500` test samples. Three outputs: per-model beeswarm, per-model mean |SHAP| bar, cross-model normalised heatmap + rank table.

### 6.3 Key design decisions

- LR and MoE use log1p(`monthly_income`); CatBoost uses raw (trees invariant to monotonic transforms)
- Soft labels computed by slicing raw numpy arrays in order — **not** via shuffled `train_loader` (shuffle misalignment bug: concatenated soft labels would not align with `X_tr_m_s` row order)
- All four models use the same 70/15/15 stratified split for fair comparison

---

## 7. `scripts/predict.py` — Production Prediction Script

### 7.1 Overview

Self-contained script that loads the saved CatBoost tuned model and produces three outputs per borrower:

1. **CatBoost default prediction** — probability + binary decisions under two threshold strategies
2. **CIBIL credit score** (300–900) — weighted component breakdown following CIBIL methodology
3. **SHAP explanation** — per-feature attribution with risk drivers, protective factors, and actionable improvement recommendations

Designed for **n8n integration** (Execute Command node) and general CLI use.

### 7.2 Input Modes

```bash
# Mode 1: Read from public Google Sheet (last row = input)
python scripts/predict.py

# Mode 2: Direct JSON input
python scripts/predict.py '{"unsecured_credit": 0.8, "age": 45, ...}'

# Mode 3: Custom Google Sheet
python scripts/predict.py --sheet-id <SHEET_ID>
```

**Google Sheet format**: Uses original Kaggle column names (`RevolvingUtilizationOfUnsecuredLines`, `NumberOfTimes90DaysLate`, etc.). The script maps them to internal names via `COL_MAP`. Default sheet ID: `1uTb3CsFbyO6TKJBkXLxNNmjhGep8NPgozKgf2GHwbBg`.

### 7.3 Preprocessing Pipeline

Same as training (Section 2): sentinel flag detection (`delinq_90 >= 96`), delinquency clipping at 10, missing indicators for `monthly_income` and `dependents`, median imputation (`MEDIAN_INCOME = 5400.0`), then `StandardScaler` transform from saved metadata.

### 7.4 CIBIL Score Calculation

Per `CIBIL_Scoring_Methodology.md`:

| Component | Weight | Formula |
| --- | --- | --- |
| Payment History | 35% | `100 - delinq_90×25 - delinq_60_89×15 - delinq_30_59×7 - default_flag×100` |
| Credit Utilization | 30% | `100 - utilization×150` |
| Credit Mix & Duration | 25% | `open_credit×2 + real_estate×10 + age_factor` |
| Other Factors | 10% | `50 + debt_penalty(-30 if ratio>0.8) + income_bonus(income/10000, max 50)` |

Final: `max(300, min(900, round(300 + weighted_score × 6)))`

Grades: Excellent (≥750), Very Good (≥700), Good (≥650), Fair (≥600), Poor (<600)

### 7.5 SHAP Explanation

Uses `shap.TreeExplainer` (exact, fast for CatBoost). Output structure:

- `feature_contributions`: all 13 features sorted by |SHAP value|, with direction
- `risk_drivers`: features pushing default probability UP, each with an actionable `recommendation` string
- `protective_factors`: features pushing default probability DOWN
- `base_value`: model's average log-odds prediction (baseline before feature effects)

### 7.6 Saved Artefacts Used

| File | Contents |
| --- | --- |
| `models/catboost_tuned.cbm` | CatBoost tuned model binary |
| `models/catboost_tuned_meta.joblib` | `StandardScaler` + `feature_cols` (13) + `thresh_f1` + `thresh_prec` |

### 7.7 Output Schema

```json
{
  "default_probability": 0.702847,
  "prediction_f1": 0,
  "prediction_conservative": 0,
  "threshold_f1": 0.7519,
  "threshold_conservative": 0.9054,
  "cibil_score": 638,
  "cibil_grade": "Fair",
  "cibil_components": {
    "payment_history": 93.0,
    "credit_utilization": 0,
    "credit_mix_duration": 72.0,
    "other_factors": 58.0
  },
  "shap_explanation": {
    "base_value": 0.068374,
    "feature_contributions": [ ... ],
    "risk_drivers": [
      {"feature": "Revolving Credit Utilization", "impact": 0.6426, "recommendation": "Reduce utilization below 30%..."}
    ],
    "protective_factors": [
      {"feature": "90+ Days Late", "impact": -0.3447}
    ]
  }
}
```

---

## 8. Cross-Notebook Reference

| Convention | Value |
| --- | --- |
| SEED | 42 everywhere (`random_state=42`, `torch.manual_seed(42)`, `np.random.seed(42)`) |
| Split | 70 / 15 / 15 stratified |
| Threshold A | Max-F1 on val set |
| Threshold B | Max recall s.t. Default-class precision ≥ 50% on val set |
| Primary metrics | ROC-AUC (ranking), Avg Precision (calibrated), F1 (classification) |
| Log-income | Only in `moe_model_with_log_income.ipynb` and `eda.ipynb` baseline LR; NOT in `boosting_models.ipynb` (trees invariant to monotonic transforms) |
| pos_weight (MoE) | 7.0 manually set; data ratio ≈14; reducing from 14→7 trades recall for precision |
| Sentinel handling | Current: `delinq_sentinel` flag (delinq_90 ≥ 96) + cap all delinq cols at 10; old version dropped sentinel rows entirely |
| Feature count | 13 features in both MoE and boosting (12 + `delinq_sentinel`); EDA LR uses 10 (no missing indicators, no sentinel) |

---

## 9. File Structure

```text
DSL_LAB/
├── Data/
│   ├── cs-training.csv               # raw dataset (~150k rows, 11 cols incl. index)
│   └── Data Dictionary.xls           # official feature descriptions
├── models/
│   ├── catboost_tuned.cbm            # CatBoost Optuna-tuned model binary
│   ├── catboost_tuned_meta.joblib    # scaler + feature_cols + thresholds
│   ├── catboost_baseline.cbm         # CatBoost baseline model binary
│   ├── catboost_baseline_meta.joblib # baseline metadata
│   └── ...                           # XGBoost, LR, MoE, KD student artefacts
├── notebooks/
│   ├── eda.ipynb                      # 40 cells: full EDA + VIF + log-transform + baseline LR
│   ├── moe_model.ipynb                # original MoE (raw monthly_income, LR KD students) — kept for reference
│   ├── moe_model_with_log_income.ipynb  # 41 cells: MoE + log1p + per-expert analysis + 3D UMAP + NN KD students
│   ├── boosting_models.ipynb          # 29 cells: XGBoost + CatBoost + Optuna
│   └── summary.ipynb                  # 46 cells: EDA + LR / MoE / KD Student / CatBoost(Optuna) comparison + SHAP
├── scripts/
│   ├── predict.py                     # Production prediction: Google Sheet → CatBoost + CIBIL + SHAP → JSON
│   ├── data_preprocessing.py          # Shared data preprocessing utilities
│   └── catboost_pipeline/             # CatBoost pipeline scripts
├── CIBIL_Scoring_Methodology.md       # CIBIL credit score formula documentation (300-900)
├── pyproject.toml                     # uv-managed dependencies
└── SESSION_SUMMARY.md                 # this file
```

**Key dependencies**: `torch`, `xgboost`, `catboost`, `scikit-learn`, `umap-learn`, `optuna`, `statsmodels`, `missingno`, `shap`, `plotly`, `pandas`, `joblib`
