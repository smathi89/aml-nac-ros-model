import pandas as pd, numpy as np, joblib

# Load features and the pairwise CSV (already contains ROS_Level_%Control)
X   = joblib.load("data/features_raw_df.pkl")            # pandas DataFrame
csv = pd.read_csv("data/ros_pairwise_pct_nonzero.csv")   # raw source for target

# Normalize Units for reliable joins
for df in (csv, X):
    if "Units" in df.columns:
        df["Units"] = df["Units"].astype(str).str.replace("μ","u",regex=False).str.strip()

# Keys that likely exist in both (we’ll auto-filter to those present)
candidate_keys = [
    "Study","Cell Line","Compound Name","Condition","Nanoparticle","Units",
    "Concentration (uM)","Concentration_mg/mL","H2O2_mM","siRNA"
]
keys = [k for k in candidate_keys if (k in X.columns) and (k in csv.columns)]
if not keys:
    raise SystemExit("No common keys to merge on. Adjust candidate_keys.")

# Make numeric columns numeric on both sides to avoid string/float mismatches
for k in ["Concentration (uM)","Concentration_mg/mL","H2O2_mM"]:
    if k in csv.columns: csv[k] = pd.to_numeric(csv[k], errors="coerce")
    if k in X.columns:   X[k]   = pd.to_numeric(X[k], errors="coerce")

# Collapse CSV to a single row per key combo (median if duplicates)
csv_unique = (csv[keys + ["ROS_Level_%Control"]]
              .groupby(keys, dropna=False, as_index=False)["ROS_Level_%Control"].median())

# Left-merge in X row order; validate right side is unique per key (many_to_one)
m = (X.reset_index(drop=False).rename(columns={"index":"_row"})[["_row"] + keys]
       .merge(csv_unique, on=keys, how="left", validate="many_to_one")
       .sort_values("_row"))

y = m["ROS_Level_%Control"].to_numpy()
missing = int(np.isnan(y).sum())
print(f"Matched rows: {len(y) - missing}/{len(y)}")
if missing:
    print("⚠️ Unmatched examples:")
    print(m.loc[m["ROS_Level_%Control"].isna(), keys].head())

# Basic stats
finite = np.isfinite(y)
print("Target stats:",
      f"n={finite.sum()}/{len(y)}",
      f"mean={float(np.nanmean(y)):.6f}",
      f"std={float(np.nanstd(y)):.6f}",
      f"min={float(np.nanmin(y)):.6f}",
      f"max={float(np.nanmax(y)):.6f}")

# Save target aligned 1:1 with X
joblib.dump(y, "data/y_target.pkl")
print("✅ wrote data/y_target.pkl (aligned with features)")
