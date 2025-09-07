import json, joblib
from pathlib import Path
p = Path("data")
meta = json.load(open(p/"target_meta.json")) if (p/"target_meta.json").exists() else {}
sch  = json.load(open(p/"feature_schema.json")) if (p/"feature_schema.json").exists() else {}
X    = joblib.load(p/"features_raw_df.pkl")
print("=== TARGET META ==="); print(json.dumps(meta, indent=2))
print("\n=== FEATURES USED (counts) ===")
cats = sch.get("categorical_used") or sch.get("categorical",[])
nums = sch.get("numeric_used") or sch.get("numeric",[])
print(f"n_samples={len(X)}, n_features={X.shape[1]}")
print(f"categorical({len(cats)}): {cats}")
print(f"numeric({len(nums)}): {nums}")
m = Path("models")/"ros_cv_metrics.json"
print("\n=== OOF METRICS ===")
print(open(m,encoding="utf-8").read() if m.exists() else "no metrics file found")
