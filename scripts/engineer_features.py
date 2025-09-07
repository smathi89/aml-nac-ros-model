import pandas as pd
import os

# === Load full cleaned dataset ===
df = pd.read_csv("data/cleaned_nac_model_data.csv")  # ⬅️ Full dataset

# === Lowercase searchable columns ===
text_cols = ["Condition", "Notes", "Source File"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower()

# === Create binary flags ===
df["PEGylated"] = df.apply(lambda row: int("peg" in " ".join([str(row[c]) for c in text_cols])), axis=1)
df["NAC_present"] = df.apply(lambda row: int("nac" in " ".join([str(row[c]) for c in text_cols])), axis=1)
df["H2O2_present"] = df.apply(lambda row: int("h2o2" in " ".join([str(row[c]) for c in text_cols])), axis=1)

# === Extract nanoparticle type from source file ===
def extract_np_type(filename):
    return filename.split(".csv")[0].split("_")[0] if isinstance(filename, str) else "unknown"

df["Nanoparticle_type"] = df["Source File"].apply(extract_np_type)

# === Save new dataset ===
output_path = "data/final_model_input.csv"
df.to_csv(output_path, index=False)

# === Preview ===
print(f"\n✅ Final dataset with engineered features saved to: {output_path}")
print(f"🧬 New columns added: PEGylated, NAC_present, H2O2_present, Nanoparticle_type")
print("\n🔍 Sample preview of new features:\n")
print(df[["PEGylated", "NAC_present", "H2O2_present", "Nanoparticle_type"]].head())


