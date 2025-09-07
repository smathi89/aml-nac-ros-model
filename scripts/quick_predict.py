import joblib, pandas as pd

X = joblib.load("data/features_raw_df.pkl")
model = joblib.load("models/ros_model_best.joblib")
pred = model.predict(X)

cols = [c for c in ["Cell Line","Compound Name","Condition","Nanoparticle","Concentration (uM)","Units"]
        if c in getattr(X, "columns", [])]
out = pd.concat([X[cols].reset_index(drop=True),
                 pd.DataFrame({"prediction": pred})], axis=1)

out.to_csv("models/predictions_best.csv", index=False)
print("wrote models/predictions_best.csv", len(pred))
