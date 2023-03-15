'''
Questo script consente di prendere più files per un certo database, e ricostruirlo sotto forma di un unico file.
È stato usato nell'esperimento riguardante il tRNA e per il 16S per ricostruire i files.
'''

import pandas as pd
import os
from pathlib import Path

# Percorso dei file excel
source_path = Path(__file__).resolve()
source_dir = source_path.parent

# Nuovo file unico da generare
path_results = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_complete/GTDB.csv"

path_ena1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/Results/ENA.csv"
path_ena2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results/ENA.csv"
#path_ena3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/ENA/2023-03-12T17_30_19.268Z.csv"
#path_ena4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/ENA/2023-03-12T17_40_24.743Z.csv"
#path_ena5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/ENA/2023-03-12T17_52_52.413Z.csv"
#path_ena6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/ENA/2023-03-12T18_03_13.908Z.csv"
#path_ena7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T09_16_09.022Z.csv"
#path_ena8 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T09_16_54.145Z.csv"
#path_ena9 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T09_17_56.461Z.csv"

path_gtdb1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/Results/GTDB.csv"
path_gtdb2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results/GTDB.csv"
path_gtdb3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/GTDB/2023-03-12T17_35_15.308Z.csv"
path_gtdb4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/GTDB/2023-03-12T17_46_28.234Z.csv"
path_gtdb5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/GTDB/2023-03-12T17_59_21.059Z.csv"
path_gtdb6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/GTDB/2023-03-12T18_06_33.468Z.csv"
#path_gtdb7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/GTDB_csv/2023-03-07T09_18_04.467Z.csv"

path_ltp1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/Results/LTP.csv"
path_ltp2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results/LTP.csv"
path_ltp3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/LTP/2023-03-12T17_32_00.280Z.csv"
path_ltp4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/LTP/2023-03-12T17_42_25.752Z.csv"
path_ltp5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/LTP/2023-03-12T17_55_03.945Z.csv"
path_ltp6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/LTP/2023-03-12T18_04_20.513Z.csv"

#Ultimi due file non utilizzati percghè non contenevano la colonna del phylum
#path_ltp6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/LTP_csv/2023-03-07T09_16_56.540Z.csv"
#path_ltp7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/LTP_csv/2023-03-07T09_17_59.197Z.csv"

path_ncbi1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/Results/NCBI.csv"
path_ncbi2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results/NCBI.csv"
path_ncbi3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/NCBI/2023-03-12T17_33_43.653Z.csv"
path_ncbi4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/NCBI/2023-03-12T17_44_26.340Z.csv"
path_ncbi5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/NCBI/2023-03-12T17_56_58.552Z.csv"
path_ncbi6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/NCBI/2023-03-12T18_05_25.955Z.csv"
#path_ncbi7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T09_16_20.407Z.csv"
#path_ncbi8 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T09_16_59.185Z.csv"
#path_ncbi9 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T09_18_01.821Z.csv"

path_silva1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/Results/SILVA.csv"
path_silva2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results/SILVA.csv"
path_silva3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/SILVA/2023-03-12T17_28_45.225Z.csv"
path_silva4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/SILVA/2023-03-12T17_38_23.880Z.csv"
path_silva5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/SILVA/2023-03-12T17_50_57.646Z.csv"
path_silva6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/SILVA/2023-03-12T18_02_06.501Z.csv"
#path_silva7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/SILVA_csv/2023-03-07T09_17_52.385Z.csv"


def rebuild_file(file_path, columns, df_results=None):
    #df = pd.read_excel(file_path)
    #df = df[df[columns[2]] != ""]
    #df = df[(df[columns[1]] >= 60) & (df[columns[1]] <= 100)]
    df = pd.read_csv(file_path)
    df = df[columns]
    if df_results is None:
        df_results = df
    else:
        df_results = pd.concat([df_results, df])
    return df_results

df_results = None
columns = ['Organism name', 'Length', 'Benchmark ID', 'Taxonomy.GTDB.domain', 'Taxonomy.GTDB.phylum']

df_results = rebuild_file(path_gtdb1, columns, df_results)
df_results = rebuild_file(path_gtdb2, columns, df_results)
#df_results = rebuild_file(path_silva3, columns, df_results)
#df_results = rebuild_file(path_silva4, columns, df_results)
#df_results = rebuild_file(path_silva5, columns, df_results)
#df_results = rebuild_file(path_silva6, columns, df_results)
#df_results = rebuild_file(path_silva7, columns, df_results)
#df_results = rebuild_file(path_ncbi8, columns, df_results)
#df_results = rebuild_file(path_ncbi9, columns, df_results)
df_results.to_csv(path_results, index=False)
'''
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
'''
'''
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
'''