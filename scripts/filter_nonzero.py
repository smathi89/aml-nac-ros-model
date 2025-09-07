# scripts/filter_nonzero.py
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/compiled_dataset.csv", help="Source CSV path")
ap.add_argument("--dst", default="data/compiled_dataset_nonzero.csv", help="Destination CSV path")
args = ap.parse_args()

src = Path(args.src)
dst = Path(args.dst)

print(f"📂 CWD: {Path.cwd()}")
print(f"🔎 Looking for source: {src}")
if not src.exists():
    print("❌ Source CSV not found. CSVs in ./data:")
    for p in Path("data").glob("*.csv"):
        print("   •", p)
    raise SystemExit(1)

df = pd.read_csv(src, encoding="utf-8")

um_candidates = [
    "Concentration (μM)", "Concentration (uM)", "Concentration (UM)",
    "Concentration_uM", "Conc_uM"
]
mgml_candidates = [
    "Concentration_mg/mL", "Concentration (mg/mL)", "Conc_mg/mL"
]

def pick_um_col():
    for c in um_candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if "μm" in cl or "um" in cl:
            return c
    return None

def pick_mg_col():
    for c in mgml_candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower().replace(" ", "")
        if "mg/ml" in cl:
            return c
    return None

um_col = pick_um_col()
mg_col = pick_mg_col()

def numify(col):
    if not col or col not in df.columns:
        return pd.Series(0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0)

mask = (numify(um_col) > 0) | (numify(mg_col) > 0)
kept = df.loc[mask].copy()

dst.parent.mkdir(parents=True, exist_ok=True)
kept.to_csv(dst, index=False, encoding="utf-8")

print(f"✅ kept {len(kept)} / {len(df)} rows")
print(f"• μM column:    {um_col or '(none found)'}")
print(f"• mg/mL column: {mg_col or '(none found)'}")
print(f"➡️ wrote: {dst}")

