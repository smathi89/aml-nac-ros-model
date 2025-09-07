import sys, re, json
import pandas as pd, numpy as np

src, aml_out, all_out = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(src)

# Coerce dose columns
for col in ['Concentration (μM)','Concentration (uM)','Concentration_mg/mL']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Unify μ/u header
if 'Concentration (μM)' in df.columns and 'Concentration (uM)' not in df.columns:
    df = df.rename(columns={'Concentration (μM)':'Concentration (uM)'})

# Normalize Units micro symbol for ASCII safety
if 'Units' in df.columns:
    df['Units'] = df['Units'].astype(str).str.replace('μ','u',regex=False).str.strip()

# Keep nonzero-dose
um = df.get('Concentration (uM)', pd.Series([np.nan]*len(df)))
mg = df.get('Concentration_mg/mL', pd.Series([np.nan]*len(df)))
mask = (um.fillna(0)>0) | (mg.fillna(0)>0)
df = df.loc[mask].copy()

# Ensure Study exists
if 'Study' not in df.columns:
    df['Study'] = 1

# Grouping keys (dose + context)
keys = [k for k in ['Study','Cell Line','Compound Name','Condition','Nanoparticle','Units','Concentration_mg/mL','Concentration (uM)'] if k in df.columns]

# Aggregate: mean for numeric EXCEPT keys, first for others
num_cols = df.select_dtypes(include=['number']).columns.tolist()
num_agg_cols = [c for c in num_cols if c not in keys]
agg = {c:'mean' for c in num_agg_cols}
for c in df.columns:
    if c not in agg and c not in keys:
        agg[c] = 'first'

df_all = df.groupby(keys, dropna=False).agg(agg).reset_index()
df_all.to_csv(all_out, index=False)

# AML subset
aml_pat = re.compile(r'(HL\s*-?\s*60|NB4|OCI\s*-?\s*AML\s*-?\s*3|THP\s*-?\s*1|MOLM\s*-?\s*13|MV4\s*-?\s*11|KG\s*-?\s*1)', re.I)
if 'Cell Line' in df_all.columns:
    df_aml = df_all[df_all['Cell Line'].astype(str).str.contains(aml_pat, na=False)].copy()
else:
    df_aml = df_all.iloc[0:0].copy()
df_aml.to_csv(aml_out, index=False)

print(f"prep done | all_rows={len(df_all)} | aml_rows={len(df_aml)}")
