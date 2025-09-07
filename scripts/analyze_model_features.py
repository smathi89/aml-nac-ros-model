import pandas as pd
import matplotlib.pyplot as plt
import joblib

# === Load trained model ===
model_path = "models/ros_model_classification.joblib"
model = joblib.load(model_path)

# === Get booster and extract feature importances ===
booster = model.get_booster()
score_dict = booster.get_score(importance_type='weight')  # Can also try 'gain' or 'cover'

# Convert to DataFrame
importance_df = pd.DataFrame({
    "Feature": list(score_dict.keys()),
    "Importance": list(score_dict.values())
}).sort_values(by="Importance", ascending=False)

# === Plot ===
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.title("XGBoost Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.grid(True)
plt.show()


