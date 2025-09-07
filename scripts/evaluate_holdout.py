import os
import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# --- safe RMSE for any sklearn version ---
def rmse_safe(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

# --- load data ---
X = joblib.load("data/X_features.pkl")
y = joblib.load("data/y_target.pkl") if os.path.exists("data/y_target.pkl") else joblib.load("data/y_labels.pkl")
if hasattr(y, "values"):
    y = y.values
y = np.ravel(y)

groups = joblib.load("data/groups.pkl") if os.path.exists("data/groups.pkl") else None
n = len(y)

# --- model pipeline (same as training) ---
k = min(30, X.shape[1])
base_pipe = Pipeline([
    ("select", SelectKBest(score_func=f_regression, k=k)),
    ("model", XGBRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", tree_method="hist",
        n_jobs=-1, random_state=42
    )),
])
log_pipe = TransformedTargetRegressor(regressor=base_pipe, func=np.log1p, inverse_func=np.expm1)

def eval_once(train_idx, test_idx, model):
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    r2  = r2_score(yte, pred)
    rmse = rmse_safe(yte, pred)
    mae = mean_absolute_error(yte, pred)
    return r2, rmse, mae, len(test_idx)

# --- repeated group-aware splits (or plain if no groups) ---
repeats = 10
test_frac_groups = 0.40  # larger test to avoid tiny test sets
rng = 42

r2s_plain, rmses_plain, maes_plain, sizes_plain = [], [], [], []
r2s_log, rmses_log, maes_log, sizes_log = [], [], [], []

if groups is not None and len(groups) == n:
    gss = GroupShuffleSplit(n_splits=repeats, test_size=test_frac_groups, random_state=rng)
    splits = list(gss.split(X, y, groups))
    split_desc = "GroupShuffleSplit"
else:
    # fallback (not group-aware)
    from sklearn.model_selection import ShuffleSplit
    ss = ShuffleSplit(n_splits=repeats, test_size=0.4, random_state=rng)
    splits = list(ss.split(X, y))
    split_desc = "ShuffleSplit"

for tr_idx, te_idx in splits:
    r2, rmse, mae, m = eval_once(tr_idx, te_idx, base_pipe)
    r2s_plain.append(r2); rmses_plain.append(rmse); maes_plain.append(mae); sizes_plain.append(m)
    r2, rmse, mae, _ = eval_once(tr_idx, te_idx, log_pipe)
    r2s_log.append(r2); rmses_log.append(rmse); maes_log.append(mae); sizes_log.append(m)

def summarize(name, r2s, rmses, maes, sizes):
    print(f"\n🔒 {name} ({split_desc}, repeats={repeats})")
    print(f"Test rows per split (min/median/max): {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")
    print(f"R²   mean={np.mean(r2s):.3f}  ± {np.std(r2s):.3f}")
    print(f"RMSE mean={np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    print(f"MAE  mean={np.mean(maes):.3f} ± {np.std(maes):.3f}")

summarize("Plain target", r2s_plain, rmses_plain, maes_plain, sizes_plain)
summarize("Log target",  r2s_log,   rmses_log,   maes_log,   sizes_log)


