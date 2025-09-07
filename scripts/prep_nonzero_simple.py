import pandas as pd, sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)

# Normalize micro symbol for safety
if "Units" in df.columns:
    df["Units"] = df["Units"].astype(str).str.replace("μ","u",regex=False).str.strip()

# Coerce potential dose columns and unify header
for col in ["Concentration (μM)","Concentration (uM)","Concentration_mg/mL"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
if "Concentration (μM)" in df.columns and "Concentration (uM)" not in df.columns:
    df = df.rename(columns={"Concentration (μM)":"Concentration (uM)"})

um = df.get("Concentration (uM)", pd.Series([np.nan]*len(df)))
mg = df.get("Concentration_mg/mL", pd.Series([np.nan]*len(df)))
mask = (um.fillna(0) > 0) | (mg.fillna(0) > 0)
df2 = df.loc[mask].copy()

df2.to_csv(dst, index=False)
print(f"wrote {dst} rows={len(df2)}")
