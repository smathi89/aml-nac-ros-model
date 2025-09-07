import joblib, numpy as np
X = joblib.load("data/features_raw_df.pkl")
y = joblib.load("data/y_target.pkl")
print("X rows:", len(X), " | y rows:", len(y))
print("y mean:", float(np.nanmean(y)), "std:", float(np.nanstd(y)),
      "min:", float(np.nanmin(y)), "max:", float(np.nanmax(y)))
