# scripts/train_ros_model.py
import os, json, pathlib
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, ShuffleSplit, cross_validate
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression

from xgboost import XGBRegressor
from scipy.stats import spearmanr

RANDOM_STATE = 42

# -------------------------- IO --------------------------
X_paths_try = ["data/features_raw_df.pkl", "data/X_features.pkl"]
y_paths_try = ["data/y_target_effect.pkl", "data/y_target.pkl", "data/y_labels.pkl"]
schema_path = "data/feature_schema.json"

X_path = next((p for p in X_paths_try if os.path.exists(p)), None)
if X_path is None:
    raise FileNotFoundError(f"Missing features file. Looked for: {', '.join(X_paths_try)}")

y_path = next((p for p in y_paths_try if os.path.exists(p)), None)
if y_path is None:
    raise FileNotFoundError(f"Missing y labels. Looked for: {', '.join(y_paths_try)}")

X = joblib.load(X_path)          # pandas DataFrame expected
y = joblib.load(y_path)
if hasattr(y, "values"):
    y = y.values
y = np.ravel(y)

if len(X) != len(y):
    raise ValueError(f"X rows ({len(X)}) != y rows ({len(y)}). Rebuild features or realign target.")

print(f"Loaded X {getattr(X, 'shape', None)}, y {y.shape} from {y_path}  |  X_path={X_path}")

# ---------------------- Helpers -------------------------
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)

def load_schema_cols(X_df: pd.DataFrame, schema_path: str):
    num_cols, cat_cols = [], []
    if os.path.exists(schema_path):
        try:
            sc = json.load(open(schema_path, "r"))
            num_cols = sc.get("numeric_used") or sc.get("numeric") or sc.get("numeric_cols") or []
            cat_cols = sc.get("categorical_used") or sc.get("categorical") or sc.get("categorical_cols") or []
        except Exception:
            pass
    num_cols = [c for c in num_cols if c in X_df.columns]
    cat_cols = [c for c in cat_cols if c in X_df.columns]
    if not num_cols:
        num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    if not cat_cols:
        cat_cols = [c for c in X_df.columns if c not in num_cols]
    return num_cols, cat_cols

# ----------------- Feature engineering ------------------
class AddDoseFeatures(BaseEstimator, TransformerMixin):
    """
    Adds:
      - dose_uM  : unified dose in μM
      - log_dose : log1p(dose_uM)
      - is_dosed : 1.0 if dose_uM > 0 else 0.0
    """
    def _norm_units(self, s: pd.Series) -> pd.Series:
        return s.astype(str).str.replace("μ", "u", regex=False).str.lower().str.strip()

    def _extract_number(self, series: pd.Series) -> pd.Series:
        s = series.astype(str).str.replace(",", "", regex=False)
        num = s.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
        return pd.to_numeric(num, errors="coerce")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Z = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()

        units_ser = None
        if "Units" in Z.columns:
            units_ser = self._norm_units(Z["Units"])

        if "Concentration (uM)" in Z.columns:
            dose = pd.to_numeric(Z["Concentration (uM)"], errors="coerce")
        else:
            dose = pd.Series(np.nan, index=Z.index, dtype="float64")

        if "Concentration (μM or mM)" in Z.columns:
            raw_num = self._extract_number(Z["Concentration (μM or mM)"])
            if units_ser is not None:
                conv = pd.Series(np.nan, index=Z.index, dtype="float64")
                mask_mm = units_ser.eq("mm")
                mask_um = units_ser.eq("um")
                mask_nm = units_ser.eq("nm")
                conv.loc[mask_mm] = raw_num.loc[mask_mm] * 1000.0
                conv.loc[mask_um] = raw_num.loc[mask_um]
                conv.loc[mask_nm] = raw_num.loc[mask_nm] / 1000.0
            else:
                conv = raw_num
            dose = dose.where(~dose.isna(), conv)

        Z["dose_uM"]  = pd.to_numeric(dose, errors="coerce")
        Z["log_dose"] = np.log1p(Z["dose_uM"])
        Z["is_dosed"] = (Z["dose_uM"] > 0).astype(float)
        return Z

# ---------------------- Preprocessor --------------------
num_cols, cat_cols = load_schema_cols(X, schema_path)
for c in ["dose_uM", "log_dose", "is_dosed"]:
    if c not in num_cols:
        num_cols.append(c)
num_cols = list(dict.fromkeys(num_cols))
cat_cols = list(dict.fromkeys(cat_cols))
print(f"Preprocessor columns → numeric={len(num_cols)}, categorical={len(cat_cols)}")

transformers = []
if num_cols:
    transformers.append(("num", Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
    ]), num_cols))
if cat_cols:
    transformers.append(("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe()),
    ]), cat_cols))

preprocessor = ColumnTransformer(transformers, remainder="drop")

# ------------------------ CV ----------------------------
def _codes(series: pd.Series) -> np.ndarray:
    return series.astype("category").cat.codes.to_numpy()

def choose_groups(X_df: pd.DataFrame, loaded_groups: np.ndarray | None):
    """
    Pick the most informative grouping with the most unique groups available.
    Order considered: ENV 'GROUP_KEY' -> Cell Line -> Study|Cell Line -> Study -> loaded file.
    """
    env_key = os.getenv("GROUP_KEY", "").strip()
    candidates: list[tuple[str, np.ndarray]] = []

    if env_key and env_key in X_df.columns:
        candidates.append((f"ENV:{env_key}", _codes(X_df[env_key])))

    if "Cell Line" in X_df.columns:
        candidates.append(("Cell Line", _codes(X_df["Cell Line"])))
    if "Study" in X_df.columns and "Cell Line" in X_df.columns:
        combo = (X_df["Study"].astype(str) + "|" + X_df["Cell Line"].astype(str))
        candidates.append(("Study|CellLine", _codes(combo)))
    if "Study" in X_df.columns:
        candidates.append(("Study", _codes(X_df["Study"])))

    if loaded_groups is not None and len(loaded_groups) == len(X_df):
        candidates.append(("loaded_groups", np.asarray(loaded_groups)))

    best_name, best_g, best_n = None, None, -1
    for name, g in candidates:
        n = int(np.unique(g).size)
        if n > best_n:
            best_name, best_g, best_n = name, g, n

    return best_name or "index", (best_g if best_g is not None else np.arange(len(X_df))), best_n

loaded_groups = None
if os.path.exists("data/groups.pkl"):
    try:
        loaded_groups = joblib.load("data/groups.pkl")
    except Exception:
        loaded_groups = None

grp_name, groups, n_unique_groups = choose_groups(X, loaded_groups)
force_cv = os.getenv("CV_MODE", "auto").lower()  # "auto" | "shuffle" | "gkf2"

if force_cv == "shuffle":
    ss = ShuffleSplit(n_splits=50, test_size=0.40, random_state=RANDOM_STATE)
    cv_splits = list(ss.split(X, y))
    cv_name = "ShuffleSplit(50, 40%)  [FORCED]"
elif force_cv == "gkf2":
    gkf = GroupKFold(n_splits=2)
    cv_splits = list(gkf.split(X, y, groups=groups))
    cv_name = f"GroupKFold(2)  [group={grp_name}]  [FORCED]"
else:
    if n_unique_groups >= 3:
        gss = GroupShuffleSplit(n_splits=30, test_size=0.40, random_state=RANDOM_STATE)
        cv_splits = list(gss.split(X, y, groups=groups))
        cv_name = f"GroupShuffleSplit(30, 40%)  [group={grp_name}]"
    elif n_unique_groups == 2:
        gkf = GroupKFold(n_splits=2)
        cv_splits = list(gkf.split(X, y, groups=groups))
        cv_name = f"GroupKFold(2)  [leave-one-group-out | group={grp_name}]"
    else:
        ss = ShuffleSplit(n_splits=50, test_size=0.40, random_state=RANDOM_STATE)
        cv_splits = list(ss.split(X, y))
        cv_name = "ShuffleSplit(50, 40%)"

fold_sizes = [len(te) for _, te in cv_splits]
print(f"Using {cv_name}. Test sizes min/median/max: {min(fold_sizes)}/{int(np.median(fold_sizes))}/{max(fold_sizes)}")

# -------------------- Robust scorers --------------------
def _clip01(a): 
    return np.clip(a, 0.0, 1.0)

def mae_clip_scorer(est, Xv, yv):
    yp = _clip01(est.predict(Xv))
    return -mean_absolute_error(yv, yp)

def rmse_clip(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def r2_skill_from_rmse(est, Xv, yv):
    yp = _clip01(est.predict(Xv))
    rmse = rmse_clip(yv, yp)
    rmse_base = rmse_clip(yv, np.full_like(yv, fill_value=np.mean(yv)))
    return 1.0 - (rmse / rmse_base)**2 if rmse_base > 0 else np.nan

def spearman_scorer(est, Xv, yv):
    yp = _clip01(est.predict(Xv))
    rho, _ = spearmanr(yv, yp)
    return 0.0 if np.isnan(rho) else float(rho)

scoring = {
    "r2_skill": r2_skill_from_rmse,         # robust alternative to classic R^2
    "neg_mse": "neg_mean_squared_error",    # unclipped MSE for reference
    "neg_mae": mae_clip_scorer,             # MAE on clipped preds
    "spearman": spearman_scorer,            # rank correlation
}

def agg(res, sizes):
    # Weighted means by test fold size
    w = np.asarray(sizes, dtype=float); w = w / w.sum()

    rmse = np.sqrt(-np.asarray(res["test_neg_mse"], dtype=float))
    mae  = -np.asarray(res["test_neg_mae"], dtype=float)

    r2s = np.asarray(res.get("test_r2_skill", np.full_like(mae, np.nan)), dtype=float)
    sp  = np.asarray(res.get("test_spearman", np.full_like(mae, np.nan)), dtype=float)

    rmse_mean = float(np.sum(rmse * w))
    rmse_std  = float(np.sqrt(np.sum(((rmse - rmse_mean)**2) * w)))
    mae_mean  = float(np.sum(mae  * w))
    mae_std   = float(np.sqrt(np.sum(((mae  - mae_mean )**2) * w)))

    return {
        "r2_skill_mean": float(np.nanmean(r2s)), "r2_skill_std": float(np.nanstd(r2s)),
        "spearman_mean": float(np.nanmean(sp)),  "spearman_std": float(np.nanstd(sp)),
        "rmse_mean": rmse_mean, "rmse_std": rmse_std,
        "mae_mean": mae_mean,   "mae_std": mae_std,
    }

# --------------------- Baseline -------------------------
def base_pipeline():
    return Pipeline([
        ("dose", AddDoseFeatures()),
        ("prep", preprocessor),
        ("var", VarianceThreshold(0.0)),
        ("model", DummyRegressor(strategy="mean")),
    ])

baseline = base_pipeline()
base_res = cross_validate(baseline, X, y, cv=cv_splits, scoring=scoring, n_jobs=1, error_score="raise")
b = agg(base_res, fold_sizes)
print("\n🧪 Baseline (predict mean w/ same preprocessing):")
print(f"CV:  {cv_name}")
print(f"R²_skill mean={b['r2_skill_mean']:.3f} ± {b['r2_skill_std']:.3f}")
print(f"Spearman  mean={np.nan:.3f} ± {np.nan:.3f}")  # Not computed for baseline easily here
print(f"RMSE mean={b['rmse_mean']:.3f} ± {b['rmse_std']:.3f}")
print(f"MAE  mean={b['mae_mean']:.3f} ± {b['mae_std']:.3f}")

# ------------------- Candidate models -------------------
def pipe(model, with_k=None):
    steps = [("dose", AddDoseFeatures()),
             ("prep", preprocessor),
             ("var", VarianceThreshold(0.0))]
    if with_k is not None:
        steps.append(("select", SelectKBest(score_func=f_regression, k=with_k)))
    steps.append(("model", model))
    return Pipeline(steps)

kmax = X.shape[1] if hasattr(X, "shape") else len(X[0])
k_candidates = [k for k in (3, 5, 8, 10) if k <= max(1, kmax)]
models = {}

# XGB (all features)
models["xgb_all"] = pipe(XGBRegressor(
    n_estimators=800, learning_rate=0.05, max_depth=3,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.25, reg_lambda=2.0, min_child_weight=3,
    objective="reg:squarederror", tree_method="hist",
    n_jobs=-1, random_state=RANDOM_STATE
))

# XGB + SelectKBest
for k in k_candidates:
    models[f"xgb_k{k}"] = pipe(XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.25, reg_lambda=2.0, min_child_weight=3,
        objective="reg:squarederror", tree_method="hist",
        n_jobs=-1, random_state=RANDOM_STATE
    ), with_k=k)

# RidgeCV (linear)
alphas = np.logspace(-3, 3, 25)
models["ridgecv"] = pipe(RidgeCV(alphas=alphas))

# RandomForest (robust for small N)
models["rf"] = pipe(RandomForestRegressor(
    n_estimators=600, max_features="sqrt",
    min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
))

# XGB optimized for MAE directly
models["xgb_mae"] = pipe(XGBRegressor(
    n_estimators=1200, learning_rate=0.03, max_depth=3,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=3.0, min_child_weight=3,
    objective="reg:absoluteerror", tree_method="hist",
    n_jobs=-1, random_state=RANDOM_STATE
))

# Optional: XGB with logit target (clip y to (0,1))
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))
def _inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))

models["xgb_logit"] = TransformedTargetRegressor(
    regressor=pipe(XGBRegressor(
        n_estimators=900, learning_rate=0.04, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.25, reg_lambda=2.0, min_child_weight=3,
        objective="reg:squarederror", tree_method="hist",
        n_jobs=-1, random_state=RANDOM_STATE
    )),
    func=_logit, inverse_func=_inv_logit, check_inverse=False
)

# -------------------- Evaluate all ----------------------
def print_with_skill(name, m, b):
    base_mae, base_rmse = b["mae_mean"], b["rmse_mean"]
    mae_skill  = 100.0 * (base_mae  - m["mae_mean"])  / base_mae  if base_mae  > 0 else 0.0
    rmse_skill = 100.0 * (base_rmse - m["rmse_mean"]) / base_rmse if base_rmse > 0 else 0.0
    print(f"\n📈 {name}  ({cv_name})")
    print(f"R²_skill mean={m['r2_skill_mean']:.3f} ± {m['r2_skill_std']:.3f}")
    print(f"Spearman  mean={m['spearman_mean']:.3f} ± {m['spearman_std']:.3f}")
    print(f"RMSE mean={m['rmse_mean']:.3f} ± {m['rmse_std']:.3f}")
    print(f"MAE  mean={m['mae_mean']:.3f} ± {m['mae_std']:.3f}")
    print(f"Skill vs baseline:  MAE {mae_skill:+.1f}%   RMSE {rmse_skill:+.1f}%")

results = {}
for name, model in models.items():
    res = cross_validate(model, X, y, cv=cv_splits, scoring=scoring, n_jobs=1, error_score="raise")
    m = agg(res, fold_sizes)
    results[name] = m
    print_with_skill(name, m, b)

# -------------------- Leaderboard -----------------------
def sort_key(item):
    m = item[1]
    return (m["mae_mean"], m["rmse_mean"], -m.get("r2_skill_mean", 0.0))

leader = sorted(results.items(), key=sort_key)
print("\n🏁 Leaderboard (by MAE, RMSE tiebreak):")
for rank, (name, m) in enumerate(leader, 1):
    print(f"{rank:>2}. {name:12s}  R²_skill={m['r2_skill_mean']:.3f}  RMSE={m['rmse_mean']:.3f}  MAE={m['mae_mean']:.3f}")

best_name, best_metrics = leader[0]
best_model = models[best_name]

# ----------------- Fit full & save ----------------------
pathlib.Path("models").mkdir(exist_ok=True)
pathlib.Path("outputs").mkdir(exist_ok=True)

best_model.fit(X, y)
best_path = "models/ros_model_best.joblib"
joblib.dump(best_model, best_path)
print(f"\n✅ Saved best model: {best_name} → {best_path}")

# Convenience variant
xgb_all_full = models["xgb_all"].fit(X, y)
joblib.dump(xgb_all_full, "models/ros_model_xgb.joblib")
print("✅ Also saved: models/ros_model_xgb.joblib")

# ----------------- Save OOF predictions for best --------
# (use the same CV splits used for evaluation)
oof = np.full_like(y, fill_value=np.nan, dtype=float)
for (tr, te) in cv_splits:
    m = clone(best_model)
    m.fit(X.iloc[tr], y[tr])
    pred = m.predict(X.iloc[te])
    oof[te] = np.clip(pred, 0.0, 1.0)

oof_mae = float(mean_absolute_error(y, oof))
oof_rmse = float(np.sqrt(mean_squared_error(y, oof)))
rho, _ = spearmanr(y, oof)
oof_spearman = 0.0 if np.isnan(rho) else float(rho)
print(f"OOF (best model)  MAE={oof_mae:.4f}  RMSE={oof_rmse:.4f}  Spearman={oof_spearman:.3f}")

oof_df = X.copy()
oof_df["y_true"] = y
oof_df["y_pred_oof"] = oof
oof_df["abs_err"] = np.abs(oof_df["y_true"] - oof_df["y_pred_oof"])
oof_path = "models/ros_oof_predictions.csv"
oof_df.to_csv(oof_path, index=False)
print(f"🧪 Saved OOF predictions → {oof_path}")

# ---------------- Calibration (Isotonic) ----------------
iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
iso.fit(oof, y)  # fit on OOF (preds vs true)
oof_cal = iso.transform(oof)
oof_mae_cal = float(mean_absolute_error(y, oof_cal))
oof_rmse_cal = float(np.sqrt(mean_squared_error(y, oof_cal)))
rho_cal, _ = spearmanr(y, oof_cal)
oof_spearman_cal = 0.0 if np.isnan(rho_cal) else float(rho_cal)

print(f"OOF (calibrated)  MAE={oof_mae_cal:.4f}  RMSE={oof_rmse_cal:.4f}  Spearman={oof_spearman_cal:.3f}")
joblib.dump(iso, "models/ros_iso_calibrator.joblib")
print("✅ Saved isotonic calibrator → models/ros_iso_calibrator.joblib")

# -------------- MAE% of range & group errors ------------
rng = float(np.nanmax(y) - np.nanmin(y))
oof_mae_pct_range = (100.0 * oof_mae / rng) if rng > 0 else np.nan
oof_mae_pct_range_cal = (100.0 * oof_mae_cal / rng) if rng > 0 else np.nan
print(f"MAE% of range (raw) = {oof_mae_pct_range:.2f}%  | (calibrated) = {oof_mae_pct_range_cal:.2f}%  (range={rng:.6f})")

group_key = "Cell Line" if "Cell Line" in oof_df.columns else None
if group_key:
    by = oof_df.groupby(group_key)["abs_err"].agg(["count", "mean", "median"]).sort_values("mean")
    by.to_csv("outputs/ros_oof_error_by_group.csv")
    print("📊 Wrote per-group OOF error → outputs/ros_oof_error_by_group.csv")

# -------------- Feature importance (tree models) --------
def dump_feature_importance(model, preprocessor, out_csv):
    try:
        # Get final feature names after ColumnTransformer & OHE
        num = []
        cat = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                num = list(cols)
            elif name == "cat":
                cat = list(cols)
        ohe = preprocessor.named_transformers_.get("cat", None)
        if ohe and hasattr(ohe, "named_steps") and "ohe" in ohe.named_steps:
            ohe = ohe.named_steps["ohe"]
            cat_names = ohe.get_feature_names_out(cat).tolist()
        else:
            cat_names = []
        feat_names = num + cat_names

        est = model.named_steps.get("model", model)
        if hasattr(est, "feature_importances_"):
            imp = pd.DataFrame({"feature": feat_names, "importance": est.feature_importances_[:len(feat_names)]})
            imp.sort_values("importance", ascending=False).to_csv(out_csv, index=False)
            print(f"🧷 Saved feature importance → {out_csv}")
    except Exception as e:
        print(f"(skip feature importance) {e}")

dump_feature_importance(best_model, preprocessor, "outputs/ros_feature_importance.csv")

# ---------------------- Report --------------------------
report = {
    "cv": cv_name,
    "group_by": grp_name,
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]) if hasattr(X, "shape") else None,
    "fold_sizes": fold_sizes,
    "baseline": b,
    "models": results,
    "best_model": {"name": best_name, **best_metrics},
    "oof": {
        "mae": oof_mae,
        "rmse": oof_rmse,
        "spearman": oof_spearman,
        "mae_pct_range": oof_mae_pct_range,
        "calibrated": {
            "mae": oof_mae_cal,
            "rmse": oof_rmse_cal,
            "spearman": oof_spearman_cal,
            "mae_pct_range": oof_mae_pct_range_cal,
        },
    },
    "paths": {
        "features": X_path,
        "labels": y_path,
        "best_model": best_path,
        "xgb_all_model": "models/ros_model_xgb.joblib",
        "oof": oof_path,
        "calibrator": "models/ros_iso_calibrator.joblib",
        "group_error": ("outputs/ros_oof_error_by_group.csv" if group_key else None),
        "feat_importance": "outputs/ros_feature_importance.csv",
    },
}
with open("outputs/ros_cv_results.json", "w") as f:
    json.dump(report, f, indent=2)
with open("models/ros_cv_metrics.json", "w") as f:
    json.dump(report, f, indent=2)
print("📝 Wrote CV summary → outputs/ros_cv_results.json (and models/ros_cv_metrics.json)")




