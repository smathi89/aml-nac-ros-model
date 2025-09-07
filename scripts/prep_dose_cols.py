import pandas as pd, numpy as np, sys
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
for col in ['Concentration (μM)', 'Concentration_mg/mL']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
if 'Units' in df.columns:
    df['Units'] = df['Units'].astype(str).str.replace('μ','u',regex=False)
um = df.get('Concentration (μM)')
mg = df.get('Concentration_mg/mL')
mask = ((um.fillna(0)>0) if um is not None else False) | ((mg.fillna(0)>0) if mg is not None else False)
df = df.loc[mask].copy()
df.to_csv(dst, index=False)
print(f"wrote {dst} rows={len(df)}")
