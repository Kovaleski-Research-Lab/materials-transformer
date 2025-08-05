import pandas as pd
import re

df = pd.read_parquet('data/IR_v09_df_all_features.parquet')
#print(f"Smiles: {df['smiles']}")

all_smiles_text = "".join(df['smiles'])

element_matches = re.findall('[A-Z][a-z]*', all_smiles_text)
unique_elements = set(element_matches)
unique_chars = set(all_smiles_text)
            
print(f"unique elements found: {sorted(list(unique_elements))}")
print(f"unique characters found: {sorted(list(unique_chars))}, Length: {len(unique_chars)}")