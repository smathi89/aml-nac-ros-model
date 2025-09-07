# quick_check_y.py
import joblib, pandas as pd
import pathlib, json

# Resolve project root (parent of "scripts")
ROOT = pathlib.Path(__file__).resolve().parents[1]
y_path = ROOT / "data" / "y_target.pkl"
meta_path = ROOT / "outputs" / "ros_cv_results.json"

print("Resolved y_path:", y_path)  # <-- helpful confirmation

y = joblib.load(y_path)

# Flatten
try:
    import numpy as np
    y_arr = y.values.ravel() if hasattr(y, "values") else np.array(y).ravel()
except Exception:
    y_arr = y

print("y length:", len(y_arr))
print("y head:", y_arr[:10])
print("y min/max:", float(min(y_arr)), float(max(y_arr)))

# Optional metadata
try:
    meta = json.load(open(meta_path))
    print("\nPaths:", meta.get("paths", {}))
    print("Best model:", meta.get("best_model", {}))
except Exception as e:
    print("\n(no ros_cv_results.json yet or parse issue)", e)

