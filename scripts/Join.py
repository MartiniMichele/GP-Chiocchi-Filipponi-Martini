
import pandas as pd
import os
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent

path_ENA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/result2/ENA_5S.xlsx"
path_GTDB_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/result2/GTDB_5S.xlsx"
path_NCBI_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/result2/NCBI_5S.xlsx"
path_SILVA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/result2/SILVA_5S.xlsx"

# Carica i file in DataFrames
df1 = pd.read_excel(path_ENA_origin)
df2 = pd.read_excel(path_GTDB_origin)
df3 = pd.read_excel(path_NCBI_origin)
df4 = pd.read_excel(path_SILVA_origin)

# Seleziona la colonna desiderata da ogni file
df1_col = df1["Benchmark ID"]
df2_col = df2["Benchmark ID"]
df3_col = df3["Benchmark ID"]
df4_col = df4["Benchmark ID"]

df_result = pd.DataFrame(columns=df1.columns)

for index, value in df1_col.iteritems():
    if value in df2_col.values and value in df3_col.values and value in df4_col.values and value:
            df_result = df_result.append(df1.loc[index], ignore_index=True)

result_df = pd.DataFrame(df_result)
result_df.to_excel("C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/fwd5s/risultato.xlsx", index=False)