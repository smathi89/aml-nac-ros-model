import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import os

# Load the dataset
df = pd.read_csv("data/final_model_input.csv")

# Replace '-' strings with NaN
df.replace("-", np.nan, inplace=True)

# Convert relevant columns to numeric (errors='coerce' turns invalid parsing into NaN)
df["H2O2 (500 µM)"] = pd.to_numeric(df["H2O2 (500 µM)"], errors='coerce')
df["NAC (µM)"] = pd.to_numeric(df["NAC (µM)"], errors='coerce')

# Clean column names if needed (optional)
df.columns = df.columns.str.strip()

# === Feature Engineering ===
df["ROS_reduced"] = df["ROS Production (Fold Change)"].astype(float) < 1.0

# Print NaN summary
print("🧼 NaN counts per feature BEFORE cleaning:")
print(df[["H2O2 (500 µM)", "NAC (µM)", "Cell Line", "PEGylated", "NAC_present", "H2O2_present", "ROS_reduced"]].isna().sum())

# Fill numeric NaNs with column means
for col in ["H2O2 (500 µM)", "NAC (µM)"]:
    df[col] = df[col].astype(float)
    df[col] = df[col].fillna(df[col].mean())

# Select features
features = [
    "H2O2 (500 µM)",
    "NAC (µM)",
    "Cell Line",
    "PEGylated",
    "NAC_present",
    "H2O2_present"
]

# Encode categorical features
df = pd.get_dummies(df, columns=["Cell Line"], drop_first=True)

X = df[[col for col in df.columns if col in features or col.startswith("Cell Line_")]]
y = df["ROS_reduced"].astype(int)

print(f"\n✅ Final feature matrix shape: {X.shape}")
print(f"✅ Final target shape: {y.shape}")

# Prevent stratify error if class count < 2
class_counts = y.value_counts()
if class_counts.min() < 2:
    stratify_option = None
    print("⚠️ Not enough samples to stratify. Proceeding without stratification.")
else:
    stratify_option = y

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=stratify_option
)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Report
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
feature_names = X.columns

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.tight_layout()

# Save plot
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/feature_importance.png")
print("✅ Feature importance plot saved to outputs/feature_importance.png")









