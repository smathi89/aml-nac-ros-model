import sys, pandas as pd, numpy as np

src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)

# Basic hygiene
if "Units" in df.columns:
    df["Units"] = df["Units"].astype(str).str.replace("μ","u",regex=False).str.strip()

for c in ["Concentration (μM)","Concentration (uM)","Concentration_mg/mL"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
if "Concentration (μM)" in df.columns and "Concentration (uM)" not in df.columns:
    df = df.rename(columns={"Concentration (μM)":"Concentration (uM)"})

# Require raw fluorescence
raw_col = None
for cand in ["ROS Measurement","ROS_Measurement","ROS Intensity","ROS_Intensity"]:
    if cand in df.columns:
        raw_col = cand; break
if raw_col is None:
    raise SystemExit("No raw ROS column like 'ROS Measurement' found.")

# Controls: prefer explicit (NAC Involved == No), else zero-dose rows, else study median
has_NAC_inv = "NAC Involved" in df.columns
conc_uM = df.get("Concentration (uM)", pd.Series([np.nan]*len(df)))
conc_mg = df.get("Concentration_mg/mL", pd.Series([np.nan]*len(df)))
is_zero_dose = (conc_uM.fillna(0)==0) & (conc_mg.fillna(0)==0)
is_explicit_ctrl = df["NAC Involved"].astype(str).str.strip().str.lower().eq("no") if has_NAC_inv else pd.Series([False]*len(df))

if "Study" not in df.columns:
    df["Study"] = "Study_1"

def pct_control_per_study(g):
    # pick baseline rows for this study
    ctrl = g[is_explicit_ctrl.loc[g.index]] if has_NAC_inv else g.iloc[0:0]
    if ctrl.empty:
        ctrl = g[is_zero_dose.loc[g.index]]
    baseline = (ctrl[raw_col].median()
                if not ctrl.empty else g[raw_col].median())
    # avoid div-by-zero
    if not np.isfinite(baseline) or baseline == 0:
        baseline = g[raw_col].replace(0, np.nan).median()
    g["ROS_Level_%Control"] = g[raw_col] / baseline
    return g

df = df.groupby("Study", dropna=False, group_keys=False).apply(pct_control_per_study)

# Light clip to keep outliers sane (0..2x control)
df["ROS_Level_%Control"] = df["ROS_Level_%Control"].clip(lower=0, upper=2)

df.to_csv(out, index=False)
print(f"wrote {out} rows={len(df)}; studies={df['Study'].nunique()}")
