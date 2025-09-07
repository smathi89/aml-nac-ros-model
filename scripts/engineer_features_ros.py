# scripts/engineer_features_ros.py
# Build features/targets to predict ROS reduction.
# Adds: target-source switch (auto|raw|pct_control), aggressive column pruning,
# and the same safety rails you already had.

import re, json, joblib, numpy as np, pandas as pd, argparse
from pathlib import Path
from typing import List, Tuple

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------- defaults (overridable by CLI) --------------------
DEFAULT_DATA_CSV = "data/combined_ros_measurements_filled.csv"
DEFAULT_RAW_TARGET_COL = "ROS Measurement"
DEFAULT_ROS_UNIT_COL   = "ROS Unit"
DEFAULT_KEEP_UNIT      = "fluorescence units"   # pass --keep-unit "" to include all
DEFAULT_SELECT_MODE    = "nac_any"              # nac_carrier | nac_any | carrier_any
DEFAULT_MIN_ROWS_STUDY = 3
DEFAULT_TARGET_SOURCE  = "auto"                 # auto | raw | pct_control

# --------------- columns that must NEVER be used as features ---------------
LEAKY_ALWAYS = [
    "% ROS+ Cells", "ROS Level", "ROS Measurement",
    "ROS Oxidation (HL60)", "ROS Oxidation (NB4)", "ROS Oxidation (OCI-AML3)",
    "ROS_Level_%Control", "__control_median__", "__is_control__", "__pct_of_control__",
    "reduction_pct_vs_control",
    # outcome/readout-ish text:
    "Observed Response", "Microscopy Images", "Observation",
    # strongly outcome-coupled numeric readouts (drop to be safe):
    "Apoptosis (%)", "Cell Viability (%)", "PARP Fold Change",
]

_MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a"}

# -------------------- helpers --------------------
def _normalize_obj(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.lower()
    s2 = s2.where(~s2.isin(_MISSING_TOKENS), np.nan)
    return s2

def exists_and_var(df, col):
    return col in df.columns and df[col].nunique(dropna=False) > 1

def any_contains(s: str, pats: List[re.Pattern]) -> bool:
    s = (s or "").lower()
    return any(p.search(s) for p in pats)

def flag_contains_any(df, cols: List[str], patterns: List[str]):
    if not cols:
        return pd.Series([False]*len(df), index=df.index)
    pats = [re.compile(p, re.I) for p in patterns]
    mat = df[cols].copy()
    for c in mat.columns:
        mat[c] = _normalize_obj(mat[c])
    flags = []
    for _, row in mat.fillna("").astype(str).iterrows():
        flags.append(any(any_contains(" ".join(row.values), pats) for _ in [0]))
    return pd.Series(flags, index=df.index)

def smart_group_keys(df: pd.DataFrame) -> Tuple[str, ...]:
    for cand in [("Study","Cell Line"), ("Source Figure/Study","Cell Line"), ("Study",), ("Source Figure/Study",)]:
        if all(c in df.columns for c in cand):
            nunique = [df[c].nunique(dropna=False) for c in cand]
            if sum(v>1 for v in nunique) >= 1:
                return cand
    return tuple()

def winsorize(s: pd.Series, lower=0.01, upper=0.99):
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)

def assign_control_baseline(df: pd.DataFrame, keys: Tuple[str, ...], raw_col: str) -> pd.DataFrame:
    df = df.copy()
    if df["__is_control__"].sum() > 0:
        if len(keys)==0:
            df["__control_median__"] = df.loc[df["__is_control__"], raw_col].median()
        else:
            ctrl = (df.loc[df["__is_control__"]]
                    .groupby(list(keys), dropna=False)[raw_col].median().reset_index(name="__ctrl__"))
            df = df.merge(ctrl, on=list(keys), how="left")
            miss = df["__ctrl__"].isna()
            if miss.any():
                df.loc[miss, "__ctrl__"] = df.loc[df["__is_control__"], raw_col].median()
            df["__control_median__"] = df["__ctrl__"].values
            df = df.drop(columns="__ctrl__")
        return df

    print("🛟 No explicit controls; using non-NAC rows as implicit controls.")
    non_nac = df.loc[~df["__has_nac__"], raw_col]
    if non_nac.notna().any():
        if len(keys)==0:
            df["__control_median__"] = non_nac.median()
        else:
            ctrl = (df.loc[~df["__has_nac__"]]
                    .groupby(list(keys), dropna=False)[raw_col].median().reset_index(name="__ctrl__"))
            df = df.merge(ctrl, on=list(keys), how="left")
            miss = df["__ctrl__"].isna()
            if miss.any():
                df.loc[miss, "__ctrl__"] = non_nac.median()
            df["__control_median__"] = df["__ctrl__"].values
            df = df.drop(columns="__ctrl__")
        return df

    print("🛟 No control-like rows; using global median of the raw target as baseline.")
    df["__control_median__"] = df[raw_col].median()
    return df

def has_observed_values(series: pd.Series, treat_obj_tokens_as_na: bool) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return series.notna().any()
    if treat_obj_tokens_as_na:
        return _normalize_obj(series).notna().any()
    return series.notna().any()

def prune_empty_constant(dfx: pd.DataFrame, cat_cols, num_cols):
    drop_cats_empty = [c for c in cat_cols if not has_observed_values(dfx[c], True)]
    drop_nums_empty = [c for c in num_cols if not has_observed_values(dfx[c], False)]
    cat_cols = [c for c in cat_cols if c not in drop_cats_empty]
    num_cols = [c for c in num_cols if c not in drop_nums_empty]
    drop_cats_const = [c for c in cat_cols if _normalize_obj(dfx[c]).nunique(dropna=True) <= 1]
    drop_nums_const = [c for c in num_cols if dfx[c].nunique(dropna=True) <= 1 or np.nanstd(dfx[c].astype(float)) == 0.0]
    cat_cols = [c for c in cat_cols if c not in drop_cats_const]
    num_cols = [c for c in num_cols if c not in drop_nums_const]
    return (cat_cols, num_cols, drop_cats_empty, drop_nums_empty, drop_cats_const, drop_nums_const)

def best_conc_col(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    for col in ["Concentration (μM)", "Concentration (μM or mM)", "Concentration_mg/mL", "H2O2_mM"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return col, s
    return "", pd.Series([np.nan]*len(df), index=df.index)

def rebuild_target_within_context(dfx: pd.DataFrame, keys: Tuple[str, ...], raw_col: str) -> np.ndarray:
    dfx = dfx.copy()
    _, conc_values = best_conc_col(dfx)
    dfx["_conc_tmp_"] = conc_values

    if len(keys)==0:
        mask0 = dfx["_conc_tmp_"] == 0
        if mask0.any():
            base = dfx.loc[mask0, raw_col].median()
        else:
            minc = dfx["_conc_tmp_"].min()
            base = dfx.loc[dfx["_conc_tmp_"] == minc, raw_col].median()
        if pd.isna(base): base = dfx[raw_col].median()
        y1 = 1.0 - (dfx[raw_col] / base)
    else:
        zero_meds = (dfx.loc[dfx["_conc_tmp_"]==0]
                     .groupby(list(keys), dropna=False)[raw_col].median().reset_index(name="__zero_med__"))
        minc = dfx.groupby(list(keys), dropna=False)["_conc_tmp_"].min().reset_index(name="__minc__")
        tmp = dfx.merge(minc, on=list(keys), how="left")
        min_meds = (tmp.loc[tmp["_conc_tmp_"]==tmp["__minc__"]]
                    .groupby(list(keys), dropna=False)[raw_col].median().reset_index(name="__min_med__"))
        base_df = zero_meds.merge(min_meds, on=list(keys), how="outer")
        base_df["__baseline__"] = base_df["__zero_med__"].where(base_df["__zero_med__"].notna(), base_df["__min_med__"])
        base_df["__baseline__"] = base_df["__baseline__"].fillna(dfx[raw_col].median())
        base_df = base_df.drop(columns=["__zero_med__","__min_med__"])
        dfx = dfx.merge(base_df, on=list(keys), how="left")
        y1 = 1.0 - (dfx[raw_col] / dfx["__baseline__"])

    y1 = pd.to_numeric(y1, errors="coerce").fillna(0.0).clip(-5,5)
    y1 = winsorize(y1, 0.01, 0.99).values
    if np.nanstd(y1) < 1e-9:
        # robust z-score
        if len(keys)==0:
            med = np.median(dfx[raw_col].values)
            mad = np.median(np.abs(dfx[raw_col].values - med)) + 1e-9
            y2 = (med - dfx[raw_col].values) / mad
        else:
            med = dfx.groupby(list(keys), dropna=False)[raw_col].transform("median")
            mad = dfx.groupby(list(keys), dropna=False)[raw_col].transform(lambda x: np.median(np.abs(x-np.median(x)))+1e-9)
            y2 = (med - dfx[raw_col]) / mad
        y2 = pd.to_numeric(y2, errors="coerce").fillna(0.0).clip(-10,10).values
        return y2
    return y1

def pct_control_target(dfx: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    col = "ROS_Level_%Control"
    if col in dfx.columns:
        v = pd.to_numeric(dfx[col], errors="coerce")
        if v.notna().sum() >= 2 and np.nanstd(v) > 0:
            y = (1.0 - (v / 100.0)).clip(-5, 5)
            y = winsorize(y, 0.01, 0.99).values
            return y, True
    return np.array([]), False

# -------------------- CLI --------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data-csv", default=DEFAULT_DATA_CSV)
ap.add_argument("--raw-target-col", default=DEFAULT_RAW_TARGET_COL)
ap.add_argument("--ros-unit-col", default=DEFAULT_ROS_UNIT_COL)
ap.add_argument("--keep-unit", default=DEFAULT_KEEP_UNIT,
               help='Exact unit to keep (case-insensitive). Pass "" to keep ALL units.')
ap.add_argument("--select-mode", default=DEFAULT_SELECT_MODE, choices=["nac_carrier","nac_any","carrier_any"])
ap.add_argument("--min-rows-per-study", type=int, default=DEFAULT_MIN_ROWS_STUDY)
ap.add_argument("--target-source", default=DEFAULT_TARGET_SOURCE, choices=["auto","raw","pct_control"])
args = ap.parse_args()

data_csv = args.data_csv
raw_target_col = args.raw_target_col
ros_unit_col   = args.ros_unit_col
keep_unit      = args.keep_unit if args.keep_unit != "" else None
SELECT_MODE    = args.select_mode
MIN_ROWS_STUDY = max(1, int(args.min_rows_per_study))
TARGET_SOURCE  = args.target_source

# -------------------- load + basic filters --------------------
df = pd.read_csv(data_csv)
print(f"📥 Loaded rows: {len(df)}")

if "Study" in df.columns and MIN_ROWS_STUDY > 1:
    vc = df["Study"].value_counts(dropna=False)
    keep_mask = df["Study"].map(vc) >= MIN_ROWS_STUDY
    before = len(df); df = df.loc[keep_mask].copy()
    print(f"🔎 Filtering small studies (<{MIN_ROWS_STUDY} rows): {before} -> {len(df)}")

if ros_unit_col in df.columns and keep_unit and (
    keep_unit.lower() in df[ros_unit_col].astype(str).str.lower().unique().tolist()
):
    before = len(df)
    df = df.loc[df[ros_unit_col].astype(str).str.lower() == keep_unit.lower()].copy()
    print(f"🔧 Filtering to single ROS Unit: {keep_unit}  ({before} -> {len(df)} rows)")
else:
    print("ℹ️ Skipping ROS unit filtering (unit col missing, desired unit not present, or disabled).")

# -------------------- flags --------------------
text_cols = [c for c in [
    "Treatment Condition","Condition","Observed Response","Compound Name","Drug",
    "Nanoparticle","Delivery Type","Concentrations Tested","Study","Source Figure/Study"
] if c in df.columns]

is_control = flag_contains_any(df, text_cols, [
    r"\bcontrol\b", r"\buntreated\b", r"\bvehicle\b", r"\bplacebo\b",
    r"\bblank\b", r"\bno\s*treatment\b", r"\bmock\b", r"\bbuffer\b",
    r"\bpbs\b", r"\bhbss\b", r"\bmedia\b", r"\b0\s*(?:µm|um|μm|uM|mM)\b",
    r"\bplga\b(?!.*nac)"
])

pat_nac = [r"\bn-?acetylcysteine\b", r"\bNAC\b"]
nac_text_cols = text_cols + [c for c in df.columns if "NAC" in str(c)]
has_nac = flag_contains_any(df, nac_text_cols, pat_nac)
# also respect boolean-ish flags if present
for col in df.columns:
    cl = str(col).strip().lower()
    if cl in {"nac involved","has_nac","nac_present"}:
        has_nac = has_nac | _normalize_obj(df[col].astype(str)).isin({"true","yes","1","y"}).fillna(False)

has_carrier = flag_contains_any(df, text_cols, [
    r"\bnano", r"\bnp[s]?\b", r"\bnanoparticle", r"\bmicelle", r"\bliposome",
    r"\bnanofiber", r"\bnanogel", r"\bnanosphere", r"\bnanocapsule", r"\bnanocarrier"
])

is_polymeric = flag_contains_any(df, text_cols, [
    r"\bplga\b", r"\bpla\b", r"\bpcl\b", r"\bpeg-?plga\b", r"\bchitosan\b",
    r"\balginate\b", r"\bdextran\b", r"\bpoly", r"\bnanofiber"
])

df["__is_control__"] = is_control & (~has_nac)
df["__has_nac__"] = has_nac
df["__has_carrier__"] = has_carrier
df["__is_polymeric__"] = is_polymeric

print(f"🧪 Control rows detected: {int(df['__is_control__'].sum())}")

# -------------------- control baseline + raw reduction --------------------
if raw_target_col not in df.columns and TARGET_SOURCE in ("auto","raw"):
    raise SystemExit(f"❌ Missing target column '{raw_target_col}' in CSV.")

keys = smart_group_keys(df)
if len(keys)==0: print("⚠️ No suitable grouping keys; will fallback to global if needed.")
df = assign_control_baseline(df, keys, raw_target_col) if raw_target_col in df.columns else df

if raw_target_col in df.columns:
    df["__pct_of_control__"] = df[raw_target_col] / df.get("__control_median__", df[raw_target_col].median())
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["__pct_of_control__"])
    df["reduction_pct_vs_control"] = winsorize((1.0 - df["__pct_of_control__"]).clip(-5,5), 0.01, 0.99)
    print(f"🎯 Built target 'reduction_pct_vs_control' (winsorized). Rows remaining: {len(df)}")

# -------------------- row selection --------------------
nac_poly = df.loc[df["__has_nac__"] & df["__has_carrier__"] & df["__is_polymeric__"]].copy()
nac_any  = df.loc[df["__has_nac__"] & df["__has_carrier__"]].copy()
only_nac = df.loc[df["__has_nac__"]].copy()
only_car = df.loc[df["__has_carrier__"]].copy()

print(f"\n🔍 Detected NAC rows: {int(df['__has_nac__'].sum())}; "
      f"polymeric rows: {int(df['__is_polymeric__'].sum())}; any-carrier rows: {len(nac_any)}")
print(f"🧪 NAC + polymeric rows: {len(nac_poly)}")

if SELECT_MODE == "nac_carrier":
    if len(nac_poly) >= 12: dfx = nac_poly
    elif len(nac_any) > 0:  dfx = nac_any; print("⚠️ Fallback: NAC + any nanocarrier")
    else:                   dfx = only_nac if len(only_nac) else only_car; print("🛟 Fallback selection applied.")
elif SELECT_MODE == "nac_any":
    dfx = only_nac if len(only_nac) else nac_any if len(nac_any) else only_car
    print("✅ SELECT_MODE=nac_any — using ANY NAC rows (carrier optional).")
else:  # carrier_any
    dfx = only_car if len(only_car) else nac_any if len(nac_any) else only_nac
    print("✅ SELECT_MODE=carrier_any — using ANY carrier rows.")

if dfx.empty:
    raise SystemExit("❌ No samples after selection.")

# -------------------- choose/build target --------------------
target_meta = {"strategy": None, "column": None,
               "select_mode": SELECT_MODE, "keep_unit": keep_unit, "min_rows_study": MIN_ROWS_STUDY}

def describe_y(y, tag):
    print(f"📊 Target stats ({tag}): n={len(y)}, mean={np.mean(y):.6f}, std={np.std(y):.6e}, min={np.min(y):.6f}, max={np.max(y):.6f}")

y = None
if TARGET_SOURCE == "pct_control":
    y_alt, ok = pct_control_target(dfx)
    if not ok:
        raise SystemExit("❌ --target-source pct_control requested but ROS_Level_%Control unavailable or constant.")
    y = y_alt; target_meta.update({"strategy":"from_ROS_Level_%Control","column":"ROS_Level_%Control"}); describe_y(y,"pct_control")

elif TARGET_SOURCE == "raw":
    if "reduction_pct_vs_control" not in dfx.columns:
        raise SystemExit("❌ raw reduction target unavailable.")
    y = dfx["reduction_pct_vs_control"].astype(float).values
    describe_y(y,"raw")
else:  # auto
    y_try = dfx.get("reduction_pct_vs_control", pd.Series(dtype=float)).astype(float).values if "reduction_pct_vs_control" in dfx.columns else None
    if y_try is not None and np.nanstd(y_try) > 1e-9 and np.nanmax(y_try) - np.nanmin(y_try) > 1e-6:
        y = y_try; target_meta.update({"strategy":"raw_measurement_vs_baseline","column":raw_target_col}); describe_y(y,"raw")
        # if raw target is skewed ≤ 0, try pct_control instead (gives learnable spread)
        if np.nanmax(y) <= 0:
            y_alt, ok = pct_control_target(dfx)
            if ok:
                print("🔁 Raw target ≤ 0 across the board — switching to ROS_Level_%Control based target.")
                y = y_alt; target_meta.update({"strategy":"from_ROS_Level_%Control","column":"ROS_Level_%Control"})
                describe_y(y,"pct_control")
    else:
        y_alt, ok = pct_control_target(dfx)
        if ok:
            y = y_alt; target_meta.update({"strategy":"from_ROS_Level_%Control","column":"ROS_Level_%Control"}); describe_y(y,"pct_control")
        else:
            # last resort: rebuild within-context
            y = rebuild_target_within_context(dfx, smart_group_keys(dfx), raw_target_col)
            target_meta.update({"strategy":"rebuilt_with_conc_baseline_or_robust_z","column":raw_target_col})
            describe_y(y,"rebuilt")

mask = np.isfinite(y)
if mask.sum() < len(y):
    dfx = dfx.loc[mask].copy()
    y = y[mask]

# -------------------- group labels BEFORE dropping IDs --------------------
group_col = None; group_values = None
for cand in ["Study","Source Figure/Study"]:
    if exists_and_var(dfx, cand):
        group_col = cand; group_values = dfx[cand].astype(str).values; break

# -------------------- feature pruning --------------------
dfx = dfx.drop(columns=[c for c in LEAKY_ALWAYS if c in dfx.columns], errors="ignore")
for idcol in ["Study","Source Figure/Study"]:
    if idcol in dfx.columns:
        dfx = dfx.drop(columns=[idcol])

if "__has_carrier__" in dfx.columns: dfx["carrier_present"] = dfx["__has_carrier__"].astype(int)
if "__is_polymeric__" in dfx.columns: dfx["polymeric_flag"]  = dfx["__is_polymeric__"].astype(int)

categorical_cols = dfx.select_dtypes(include=["object"]).columns.tolist()
numeric_cols     = dfx.select_dtypes(include=["number"]).columns.tolist()

(cat_ok, num_ok, dce, dne, dcc, dnc) = prune_empty_constant(dfx, categorical_cols, numeric_cols)
if dce: print(f"🧽 Dropping empty categorical columns: {dce}")
if dne: print(f"🧽 Dropping empty numeric columns: {dne}")
if dcc: print(f"🧽 Dropping constant categorical columns: {dcc}")
if dnc: print(f"🧽 Dropping constant numeric columns: {dnc}")

categorical_cols = cat_ok; numeric_cols = num_ok

print(f"\n🧾 Final feature columns: {list(dfx.columns)}")
print("Categorical (used):", categorical_cols)
print("Numeric (used):", numeric_cols)

# (optional) preview transform for sanity
numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer([("num", numeric_transformer, numeric_cols), ("cat", categorical_transformer, categorical_cols)], remainder="drop")
_ = preprocessor.fit_transform(dfx)

# -------------------- save --------------------
Path("data").mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True)

joblib.dump(dfx, "data/features_raw_df.pkl")
with open("data/feature_schema.json","w") as f:
    json.dump({"categorical_cols": categorical_cols, "numeric_cols": numeric_cols}, f, indent=2)
joblib.dump(y, "data/y_target.pkl")
with open("data/target_meta.json","w") as f:
    json.dump(target_meta, f, indent=2)

if group_col is not None and group_values is not None and len(np.unique(group_values)) >= 2:
    joblib.dump(group_values, "data/groups.pkl")
    with open("data/groups_meta.pkl","w") as f:
        json.dump({"group_col": group_col}, f)
    print(f"👥 Using group column for CV: {group_col} (unique groups: {len(np.unique(group_values))})")
else:
    print("👥 No usable group column; skipping groups.pkl")

print("\n💾 Saved:\n - data/features_raw_df.pkl (+ feature_schema.json)\n - data/y_target.pkl (+ target_meta.json)\n - data/groups.pkl (if available)")









