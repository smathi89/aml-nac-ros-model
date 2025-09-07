import pandas as pd
import os

# === Load the engineered dataset ===
input_path = "data/final_model_input.csv"
df = pd.read_csv(input_path)

# === Clean and convert fold change to float ===
def to_float(val):
    try:
        val = str(val).replace("~", "").replace("%", "").strip()
        val = val.split("±")[0].strip()  # Keep only before ± if present
        return float(val)
    except:
        return None

df["ROS_FC_clean"] = df["ROS Production (Fold Change)"].apply(to_float)

# === Add binary classification column ===
# 1 = ROS reduced (Fold Change < 1), 0 = not reduced
df["ROS_reduced"] = df["ROS_FC_clean"].apply(lambda x: 1 if x is not None and x < 1.0 else 0)

# === Save to new CSV ===
output_path = "data/nac_peg_binary_classification.csv"
df.to_csv(output_path, index=False)

# === Preview result ===
print(f"\n✅ Classification-ready dataset saved to: {output_path}")
print(df[["ROS Production (Fold Change)", "ROS_FC_clean", "ROS_reduced"]].head())


