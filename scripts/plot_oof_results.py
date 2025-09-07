# scripts/plot_oof_results.py
import pathlib
import math
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = pathlib.Path(__file__).resolve().parents[1]
OOF_PATH = ROOT / "models" / "ros_oof_predictions.csv"
CAL_PATH = ROOT / "models" / "ros_iso_calibrator.joblib"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

def rmse(a, b):
    return float(math.sqrt(mean_squared_error(a, b)))

def spearman(a, b):
    rho, _ = spearmanr(a, b)
    return float(rho) if not (rho is None or math.isnan(rho)) else float("nan")

def metric_block(y_true, y_pred, label):
    m = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
    }
    print(f"[{label}] MAE={m['mae']:.6f}  RMSE={m['rmse']:.6f}  Spearman={m['spearman']:.3f}")
    return m

def scatter_true_vs_pred(df, y_true_col, y_pred_col, out_png, title):
    x = df[y_true_col].to_numpy()
    y = df[y_pred_col].to_numpy()
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, alpha=0.7, edgecolor="none")
    plt.plot([lo, hi], [lo, hi], linestyle="--")  # ideal line
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def residuals_plot(df, y_true_col, y_pred_col, out_png, title):
    r = df[y_true_col] - df[y_pred_col]
    plt.figure(figsize=(6,4))
    plt.scatter(df[y_true_col], r, alpha=0.7, edgecolor="none")
    plt.axhline(0, linestyle="--")
    plt.xlabel("True values")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def residual_hist(df, y_true_col, y_pred_col, out_png, title):
    r = df[y_true_col] - df[y_pred_col]
    plt.figure(figsize=(6,4))
    plt.hist(r, bins=12, edgecolor="black", alpha=0.8)
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def group_boxplot(df, group_col, err_col, out_png, title):
    # Sorted by median error
    order = (
        df.groupby(group_col)[err_col]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )
    # Matplotlib boxplot by group
    data = [df.loc[df[group_col] == g, err_col].values for g in order]
    plt.figure(figsize=(max(6, 0.5*len(order)+2), 5))
    plt.boxplot(data, labels=order, showfliers=False)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(err_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def main():
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"Missing {OOF_PATH}. Run training first to produce OOF predictions.")

    df = pd.read_csv(OOF_PATH)
    if "y_true" not in df.columns or "y_pred_oof" not in df.columns:
        raise ValueError("ros_oof_predictions.csv must contain 'y_true' and 'y_pred_oof' columns.")

    # Clip to [0,1] for safety (should already be close)
    df["y_true"] = df["y_true"].clip(0, 1)
    df["y_pred_oof"] = df["y_pred_oof"].clip(0, 1)

    print(f"Loaded OOF: {OOF_PATH}  shape={df.shape}")

    # Metrics: RAW
    metrics_raw = metric_block(df["y_true"], df["y_pred_oof"], "RAW")

    # Try calibrator
    has_cal = CAL_PATH.exists()
    if has_cal:
        iso = joblib.load(CAL_PATH)
        df["y_pred_oof_cal"] = np.asarray(iso.transform(df["y_pred_oof"].to_numpy()), dtype=float).clip(0, 1)
        metrics_cal = metric_block(df["y_true"], df["y_pred_oof_cal"], "CALIBRATED")
    else:
        print(f"(no calibrator found at {CAL_PATH}; skipping calibrated plots)")
        metrics_cal = None

    # Save a tiny metrics JSON for convenience
    summary = {"raw": metrics_raw}
    if metrics_cal is not None:
        summary["calibrated"] = metrics_cal
    json_path = OUT_DIR / "oof_plot_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics → {json_path}")

    # Plots (RAW)
    scatter_true_vs_pred(df, "y_true", "y_pred_oof",
                         OUT_DIR / "oof_scatter_raw.png",
                         "OOF: Predicted vs True (raw)")
    residuals_plot(df, "y_true", "y_pred_oof",
                   OUT_DIR / "oof_residuals_raw.png",
                   "OOF Residuals (raw)")
    residual_hist(df, "y_true", "y_pred_oof",
                  OUT_DIR / "oof_residual_hist_raw.png",
                  "OOF Residuals Histogram (raw)")

    # Plots (CALIBRATED) if available
    if has_cal:
        scatter_true_vs_pred(df, "y_true", "y_pred_oof_cal",
                             OUT_DIR / "oof_scatter_cal.png",
                             "OOF: Predicted vs True (calibrated)")
        residuals_plot(df, "y_true", "y_pred_oof_cal",
                       OUT_DIR / "oof_residuals_cal.png",
                       "OOF Residuals (calibrated)")
        residual_hist(df, "y_true", "y_pred_oof_cal",
                      OUT_DIR / "oof_residual_hist_cal.png",
                      "OOF Residuals Histogram (calibrated)")

    # Optional: per-group error boxplot if a grouping column exists
    if "Cell Line" in df.columns:
        df["abs_err_raw"] = (df["y_true"] - df["y_pred_oof"]).abs()
        group_boxplot(df, "Cell Line", "abs_err_raw",
                      OUT_DIR / "oof_abs_error_by_cell_line_raw.png",
                      "Absolute Error by Cell Line (raw)")

        if has_cal:
            df["abs_err_cal"] = (df["y_true"] - df["y_pred_oof_cal"]).abs()
            group_boxplot(df, "Cell Line", "abs_err_cal",
                          OUT_DIR / "oof_abs_error_by_cell_line_cal.png",
                          "Absolute Error by Cell Line (calibrated)")

    print("Saved plots to:", OUT_DIR)

if __name__ == "__main__":
    main()
