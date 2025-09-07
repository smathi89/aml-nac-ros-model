import sys, pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)

# Normalize units & doses
if "Units" in df.columns:
    df["Units"] = df["Units"].astype(str).str.replace("μ","u",regex=False).str.strip()
for c in ["Concentration (μM)","Concentration (uM)","Concentration_mg/mL"]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
if "Concentration (μM)" in df.columns and "Concentration (uM)" not in df.columns:
    df = df.rename(columns={"Concentration (μM)":"Concentration (uM)"})
if "Study" not in df.columns:
    df["Study"] = "Study_1"

# Find a RAW ROS column
raw_col = None
for cand in ["ROS Measurement","ROS_Measurement","ROS Intensity","ROS_Intensity"]:
    if cand in df.columns: raw_col = cand; break
if raw_col is None:
    raise SystemExit("No raw ROS column like 'ROS Measurement'/'ROS Intensity' found.")

# Helpers for controls
conc_uM = df.get("Concentration (uM)", pd.Series([np.nan]*len(df)))
conc_mg = df.get("Concentration_mg/mL", pd.Series([np.nan]*len(df)))
is_zero = (conc_uM.fillna(0)==0) & (conc_mg.fillna(0)==0)
has_NAC = "NAC Involved" in df.columns
is_nonNAC = df["NAC Involved"].astype(str).str.strip().str.lower().eq("no") if has_NAC else pd.Series([False]*len(df))

def per_study_pct(g):
    ctrl = g[is_nonNAC.loc[g.index]]
    if ctrl.empty:
        ctrl = g[is_zero.loc[g.index]]
    baseline = (ctrl[raw_col].median() if not ctrl.empty else g[raw_col].median())
    if not np.isfinite(baseline) or baseline == 0:
        baseline = g[raw_col].replace(0,np.nan).median()
    g["ROS_Level_%Control"] = g[raw_col] / baseline
    return g

df = df.groupby("Study", dropna=False, group_keys=False).apply(per_study_pct)
# If your target is too flat, you can relax/remove this clip later
df["ROS_Level_%Control"] = df["ROS_Level_%Control"].clip(0, 2)

# Keep ONLY non-zero dose rows for training
mask = (conc_uM.fillna(0)>0) | (conc_mg.fillna(0)>0)
df = df.loc[mask].copy()

df.to_csv(out, index=False)
print(f"wrote {out} rows={len(df)}; studies={df['Study'].nunique()}")
