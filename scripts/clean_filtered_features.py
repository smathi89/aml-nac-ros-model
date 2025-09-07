import pandas as pd
import os
import re

# === Load filtered dataset ===
df = pd.read_csv("data/filtered_nac_peg_only.csv")

# === Define a function to clean common string formats ===
def clean_numeric(val):
    if pd.isna(val):
        return None
    val = str(val)
    val = val.replace("~", "")           # Remove ~
    val = val.replace("%", "")           # Remove %
    val = val.split("±")[0].strip()      # Keep value before ±
    try:
        return float(val)
    except:
        return None

# === Identify likely numeric columns to clean ===
likely_numeric_cols = []
for col in df.columns:
    if df[col].dtype == object:
        sample = df[col].dropna().astype(str).head(10)
        if sample.str.contains(r"\d").any():  # contains digits
            likely_numeric_cols.append(col)

# === Apply the cleaner ===
for col in likely_numeric_cols:
    df[col + "_cleaned"] = df[col].apply(clean_numeric)

# === Drop original messy columns (optional, or keep both) ===
# df = df.drop(columns=likely_numeric_cols)

# === Save cleaned dataset ===
output_path = "data/filtered_cleaned_nac_peg_only.csv"
df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")
print(f"Columns cleaned: {likely_numeric_cols}")
