import joblib, pathlib
p = pathlib.Path("data")
X = joblib.load(p/"features_raw_df.pkl")
print("Samples:", len(X))
