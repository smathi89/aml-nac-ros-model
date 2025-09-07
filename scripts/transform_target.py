import numpy as np, joblib
y = joblib.load("data/y_target.pkl")
# Choose ONE:
# y = y - 1.0                 # delta
y = np.log(np.clip(y, 1e-8, None))   # log-ratio
joblib.dump(y, "data/y_target.pkl")
print("Transformed y saved. mean=", float(np.nanmean(y)), "std=", float(np.nanstd(y)))
