# scripts/predict_ros.py
# -*- coding: utf-8 -*-
import argparse, sys, re
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ros_model.joblib", help="Path to saved joblib pipeline")
    ap.add_argument("--features", default="data/features_raw_df.pkl", help="PKL of engineered feature DataFrame")
    ap.add_argument("--out", default=None, help="Output CSV (default: models/predictions_<modelname>.csv)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    m_path = Path(args.model); X_path = Path(args.features)
    if not m_path.exists():
        print(f"Model not found: {m_path}"); sys.exit(1)
    if not X_path.exists():
        print(f"Features not found: {X_path}"); sys.exit(1)

    print(f"Loading model: {m_path}")
    model = joblib.load(m_path)
    print(f"Loading features: {X_path}")
    X = joblib.load(X_path)

    print(f"Features shape: {getattr(X, 'shape', 'unknown')}")
    try:
        pred = model.predict(X)
    except Exception as e:
        print("Prediction failed:", repr(e)); sys.exit(1)

    out_path = Path(args.out) if args.out else Path("models") / f"predictions_{m_path.stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ID columns if present (ASCII only; detect concentration uM flexibly)
    id_candidates = ["Study","Cell Line","Compound Name","Condition","Nanoparticle","Concentration_mg/mL","Units"]
    id_cols = [c for c in id_candidates if hasattr(X, "columns") and c in X.columns]

    def find_conc_um(cols):
        for col in cols:
            s = str(col).lower().replace("μ","u").replace("µ","u")
            if "concentration" in s and re.search(r"\bu?m\b", s):
                return col
        return None

    if hasattr(X, "columns"):
        ccol = find_conc_um(X.columns)
        if ccol and ccol not in id_cols:
            id_cols.append(ccol)

    df_out = pd.DataFrame({"prediction": np.asarray(pred)})
    if id_cols:
        df_out = pd.concat([X[id_cols].reset_index(drop=True), df_out], axis=1)

    df_out.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}  (n={len(df_out)})")

    if args.verbose:
        print(df_out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()


