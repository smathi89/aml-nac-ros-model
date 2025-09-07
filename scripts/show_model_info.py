import json, joblib, os, pandas as pd
from pathlib import Path

p = Path("data")
meta = json.load(open(p/"target_meta.json", "r")) if (p/"target_meta.json").exists() else {}
sch  = json.load(open(p/"feature_schema.json","r")) if (p/"feature_schema.json").exists() else {}
X    = joblib.load(p/"features_raw_df.pkl")
groups_path = p/"groups.pkl"
groups = joblib.load(groups_path) if groups_path.exists() else None

print("=== TARGET META ===")
print(json.dumps(meta, indent=2))
print("\n=== OOF METRICS ===")
mfile = Path("models")/"ros_cv_metrics.json"
if mfile.exists():
    print(open(mfile,"r",encoding="utf-8").read())
else:
    print("ros_cv_metrics.json not found")

print("\n=== FEATURES USED ===")
print(f"n_samples={len(X)}, n_features={X.shape[1]}")
if sch:
    cats = sch.get("categorical_used") or sch.get("categorical",[])
    nums = sch.get("numeric_used") or sch.get("numeric",[])
    print(f"categorical({len(cats)}):", cats)
    print(f"numeric({len(nums)}):", nums)
else:
    print("Columns:", list(X.columns)[:40], "...")

if groups is not None:
    try:
        import numpy as np
        print("\n=== GROUPS (Study) ===")
        g = pd.Series(groups, name="Study")
        print("unique groups:", g.nunique())
        print(g.value_counts())
    except Exception as e:
        print("groups summary failed:", e)
