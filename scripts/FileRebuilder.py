'''
Questo script consente di prendere più files per un certo database, e ricostruirlo sotto forma di un unico file.
È stato usato nell'esperimento riguardante il tRNA per ricostruire i files.
'''

import pandas as pd
import os
from pathlib import Path

# Percorso dei files excels
source_path = Path(__file__).resolve()
source_dir = source_path.parent
path_results = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Results/GeneralFile.xlsx"
path_DB_archaea = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Archaea/2023-02-10T16_45_31.972Z.csv"
path_DB_bacteria1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Bacteria/LTP/2023-02-10T17_55_00.807Z.csv"
path_DB_bacteria2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Bacteria/LTP/2023-02-10T17_57_51.370Z.csv"
path_DB_bacteria3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Bacteria/LTP/2023-02-10T17_58_56.355Z.csv"
path_DB_bacteria4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Bacteria/LTP/2023-02-10T17_59_45.338Z.csv"
path_DB_bacteria5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Bacteria/LTP/2023-02-10T18_01_42.674Z.csv"
path_DB_bacteria6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Bacteria/LTP/2023-02-10T18_13_54.521Z.csv"
path_DB_bacteria7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Bacteria/ENA/2023-02-10T17_14_19.426Z.csv"
path_DB_bacteria8 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Bacteria/ENA/2023-02-10T17_16_28.502Z.csv"
path_DB_bacteria9 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Bacteria/ENA/2023-02-10T17_18_44.474Z.csv"
path_DB_eukaryota = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Eukaryota/2023-02-10T18_36_29.183Z.csv"
path_ENA = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Results/ENA.xlsx"
path_GTDB = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Results/GTDB.xlsx"
path_LTP = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Results/LTP.xlsx"
path_NCBI = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Results/NCBI.xlsx"
path_SILVA = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/Results/SILVA.xlsx"

def rebuild_file(file_path, columns, df_results=None):
    df = pd.read_excel(file_path)
    #df = df[df[columns[2]] != ""]
    #df = df[(df[columns[1]] >= 60) & (df[columns[1]] <= 100)]
    df = df[columns]
    if df_results is None:
        df_results = df
    else:
        df_results = pd.concat([df_results, df])
    return df_results

df_results = None
columns = ['Organism name', 'Length', 'Benchmark ID'] #, 'Taxonomy.LTP.superkingdom', 'Taxonomy.LTP.class', 'Taxonomy.LTP.order']
df_results = rebuild_file(path_ENA, columns, df_results)
df_results = rebuild_file(path_GTDB, columns, df_results)
df_results = rebuild_file(path_LTP, columns, df_results)
df_results = rebuild_file(path_NCBI, columns, df_results)
df_results = rebuild_file(path_SILVA, columns, df_results)
#df_results = rebuild_file(path_DB_bacteria5, columns, df_results)
#+df_results = rebuild_file(path_DB_bacteria6, columns, df_results)
#df_results = rebuild_file(path_DB_bacteria7, columns, df_results)
#df_results = rebuild_file(path_DB_bacteria8, columns, df_results)
#df_results = rebuild_file(path_DB_bacteria9, columns, df_results)
#df_results = rebuild_file(path_DB_eukaryota, columns, df_results)
df_results.to_excel(path_results, index=False)