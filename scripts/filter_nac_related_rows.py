import pandas as pd
import os

# Load the full cleaned dataset
df = pd.read_csv("data/cleaned_nac_model_data.csv")

# Define case-insensitive keywords
keywords = ["nac", "peg", "pn", "nanoparticle", "encapsulation"]

# Columns to check
columns_to_check = ["Condition", "Notes", "Source File"]

# Function to check for keyword match
def row_matches(row):
    for col in columns_to_check:
        val = str(row[col]) if col in row and pd.notna(row[col]) else ""
        if any(kw in val.lower() for kw in keywords):
            return True
    return False

# Filter the rows
filtered_df = df[df.apply(row_matches, axis=1)]

# Save the new filtered dataset
output_path = "data/filtered_nac_peg_only_v2.csv"
filtered_df.to_csv(output_path, index=False)

print(f"✅ Filtered dataset saved to: {output_path}")
print(f"📊 Rows before: {len(df)}, after filtering: {len(filtered_df)}")


