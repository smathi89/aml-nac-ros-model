import joblib, json, pathlib
p=pathlib.Path("data")
X=joblib.load(p/"features_raw_df.pkl")
m=json.load(open(p/"target_meta.json"))
print("Samples:", len(X), "| target_meta keys:", list(m.keys()))
