'''
Questo script consente di effettuare un join per due o più file excel che si vogliono prendere in esame.
È stato usato per l'esperimento riguardante il tRNA, dove si andava a verificare se in tutti i file considerati era
presente l'id della molecola. Quindi, se la verifica era positiva, allora si aggiungeva la riga (contenente la colonna
del nome del file, dell'identificativo, del superkingdom e del phylum) in un altro file excel.
Altrimenti saltava al prossimo id della molecola da confrontare.
'''
import pandas as pd
import os
from pathlib import Path

# Percorsi dei files excel
source_path = Path(__file__).resolve()
source_dir = source_path.parent

path_ENA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results_phylum_noduplicates/ENA_phylum_noduplicates.xlsx"
path_GTDB_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results_phylum_noduplicates/GTDB_phylum_noduplicates.xlsx"
path_LTP_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results_phylum_noduplicates/LTP_phylum_noduplicates.xlsx"
path_NCBI_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results_phylum_noduplicates/NCBI_phylum_noduplicates.xlsx"
path_SILVA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results_phylum_noduplicates/SILVA_phylum_noduplicates.xlsx"

# Carica i file in DataFrames
df1 = pd.read_excel(path_ENA_origin)
df2 = pd.read_excel(path_GTDB_origin)
df3 = pd.read_excel(path_LTP_origin)
df4 = pd.read_excel(path_NCBI_origin)
df5 = pd.read_excel(path_SILVA_origin)

# Seleziona la colonna desiderata da ogni file
df1_col = df1["Benchmark ID"]
df2_col = df2["Benchmark ID"]
df3_col = df3["Benchmark ID"]
df4_col = df4["Benchmark ID"]
df5_col = df5["Benchmark ID"]

df_result = pd.DataFrame(columns=df1.columns)

for index, value in df1_col.iteritems():
    if value in df2_col.values and value in df5_col.values: #and value in df4_col.values and value in df5_col.values and value :
        df_result = df_result.append(df1.loc[index], ignore_index=True)

result_df = pd.DataFrame(df_result)
result_df.to_excel("C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/16S_csv/Results_phylum_noduplicates/join_phylum_noduplicates.xlsx", index=False)