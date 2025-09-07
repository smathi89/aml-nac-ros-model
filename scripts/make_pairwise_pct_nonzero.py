import sys, pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)

# hygiene
if "Units" in df.columns:
    df["Units"] = df["Units"].astype(str).str.replace("μ","u",regex=False).str.strip()

for c in ["Concentration (μM)","Concentration (uM)","Concentration_mg/mL"]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
if "Concentration (μM)" in df.columns and "Concentration (uM)" not in df.columns:
    df = df.rename(columns={"Concentration (μM)":"Concentration (uM)"})
if "Study" not in df.columns:
    df["Study"] = "Study_1"

# unify dose into uM from "Concentration (uM)" or "Concentration (μM or mM)"+Units
df["Concentration (uM)"] = df.get("Concentration (uM)", np.nan)
if "Concentration (μM or mM)" in df.columns:
    raw = pd.to_numeric(df["Concentration (μM or mM)"], errors="coerce")
    units = df.get("Units","").astype(str).str.lower().str.strip()
    conv = np.where(units.eq("mm"), raw*1000.0,
           np.where(units.eq("um"), raw,
           np.where(units.eq("nm"), raw/1000.0, np.nan)))
    need = df["Concentration (uM)"].isna() | (df["Concentration (uM)"]==0)
    df.loc[need, "Concentration (uM)"] = conv[need]

# raw ROS column
raw_col = None
for cand in ["ROS Measurement","ROS_Measurement","ROS Intensity","ROS_Intensity"]:
    if cand in df.columns: raw_col = cand; break
if raw_col is None:
    raise SystemExit("No raw ROS column like 'ROS Measurement' or 'ROS Intensity' found.")

# helpers
conc_uM = df["Concentration (uM)"]
conc_mg = df.get("Concentration_mg/mL", pd.Series([np.nan]*len(df)))
is_zero = (conc_uM.fillna(0)==0) & (conc_mg.fillna(0)==0)
has_NAC = "NAC Involved" in df.columns
is_nonNAC = df["NAC Involved"].astype(str).str.strip().str.lower().eq("no") if has_NAC else pd.Series([False]*len(df))

# include Condition so baselines are matched more tightly when present
keys = ["Study","Cell Line","Condition","Compound Name","Nanoparticle"]
for k in keys:
    if k not in df.columns: df[k] = "UNK"

def choose_baseline(g):
    s  = g["Study"].iloc[0]
    cl = g["Cell Line"].iloc[0]
    # Strictest -> broadest (never use pair-median fallback)
    candidates = [
        g[is_nonNAC.loc[g.index]],                                 # non-NAC inside pair
        g[is_zero.loc[g.index]],                                   # zero-dose inside pair
        df[(df["Study"].eq(s)) & (df["Cell Line"].eq(cl)) & is_nonNAC],  # non-NAC same study+cell
        df[(df["Study"].eq(s)) & (df["Cell Line"].eq(cl)) & is_zero],    # zero-dose same study+cell
        df[df["Study"].eq(s) & is_nonNAC],                         # non-NAC same study
        df[df["Study"].eq(s) & is_zero],                           # zero-dose same study
        df[is_nonNAC],                                             # any non-NAC
        df[is_zero],                                               # any zero-dose
    ]
    for ctrl in candidates:
        if not ctrl.empty:
            b = ctrl[raw_col].median()
            if np.isfinite(b) and b != 0:
                return b
    # Last resort: global median of non-zeros
    return df[raw_col].replace(0, np.nan).median()

def make_pct(g):
    baseline = choose_baseline(g)
    g["ROS_Level_%Control"] = g[raw_col] / baseline
    return g

df = df.groupby(keys, dropna=False, group_keys=False).apply(make_pct)

# don't clip (preserve all variance); keep strictly positive
df["ROS_Level_%Control"] = df["ROS_Level_%Control"].replace([np.inf, -np.inf], np.nan).dropna()
df = df[df["ROS_Level_%Control"].notna()]

# keep non-zero dose rows only
mask = (conc_uM.fillna(0)>0) | (conc_mg.fillna(0)>0)
df = df.loc[mask].copy()

# ensure uM not all-NaN (quiet SimpleImputer)
df["Concentration (uM)"] = df.groupby("Study")["Concentration (uM)"].transform(lambda s: s.fillna(0.0))

df.to_csv(out, index=False)
print(f"wrote {out} rows={len(df)}; studies={df['Study'].nunique()}")
