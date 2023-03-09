'''
Questo script consente di prendere più files per un certo database, e ricostruirlo sotto forma di un unico file.
È stato usato nell'esperimento riguardante il tRNA per ricostruire i files.
'''

import pandas as pd
import os
from pathlib import Path

# Percorso dei file excel
source_path = Path(__file__).resolve()
source_dir = source_path.parent

# Nuovo file unico da generare
path_results = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nH_16S_csv/Results/SILVA.csv"

path_ena1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T08_53_53.450Z.csv"
path_ena2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T08_54_12.291Z.csv"
path_ena3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T08_57_58.904Z.csv"
path_ena4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T09_02_42.369Z.csv"
path_ena5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T09_08_15.245Z.csv"
path_ena6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T09_13_35.055Z.csv"
path_ena7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T09_16_09.022Z.csv"
path_ena8 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T09_16_54.145Z.csv"
path_ena9 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/ENA_csv/2023-03-07T09_17_56.461Z.csv"

path_gtdb1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/GTDB_csv/2023-03-07T08_59_23.909Z.csv"
path_gtdb2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/GTDB_csv/2023-03-07T09_05_04.729Z.csv"
path_gtdb3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/GTDB_csv/2023-03-07T09_11_09.577Z.csv"
path_gtdb4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/GTDB_csv/2023-03-07T09_15_23.302Z.csv"
path_gtdb5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/GTDB_csv/2023-03-07T09_16_24.989Z.csv"
path_gtdb6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/GTDB_csv/2023-03-07T09_17_01.713Z.csv"
path_gtdb7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/GTDB_csv/2023-03-07T09_18_04.467Z.csv"

path_ltp1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/LTP_csv/2023-03-07T08_58_33.521Z.csv"
path_ltp2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/LTP_csv/2023-03-07T09_03_33.787Z.csv"
path_ltp3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/LTP_csv/2023-03-07T09_09_19.457Z.csv"
path_ltp4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/LTP_csv/2023-03-07T09_14_02.691Z.csv"
path_ltp5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/LTP_csv/2023-03-07T09_16_14.262Z.csv"
#Ultimi due file non utilizzati percghè non contenevano la colonna del phylum
path_ltp6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/LTP_csv/2023-03-07T09_16_56.540Z.csv"
path_ltp7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/LTP_csv/2023-03-07T09_17_59.197Z.csv"

path_ncbi1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T08_54_45.380Z.csv"
path_ncbi2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T08_55_04.741Z.csv"
path_ncbi3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T08_58_59.890Z.csv"
path_ncbi4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T09_04_20.912Z.csv"
path_ncbi5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T09_10_01.210Z.csv"
path_ncbi6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T09_14_44.722Z.csv"
path_ncbi7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T09_16_20.407Z.csv"
path_ncbi8 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T09_16_59.185Z.csv"
path_ncbi9 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/NCBI_csv/2023-03-07T09_18_01.821Z.csv"

path_silva1 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/SILVA_csv/2023-03-07T08_57_31.605Z.csv"
path_silva2 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/SILVA_csv/2023-03-07T09_01_53.913Z.csv"
path_silva3 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/SILVA_csv/2023-03-07T09_07_32.664Z.csv"
path_silva4 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/SILVA_csv/2023-03-07T09_13_06.342Z.csv"
path_silva5 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/SILVA_csv/2023-03-07T09_16_01.710Z.csv"
path_silva6 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/SILVA_csv/2023-03-07T09_16_51.115Z.csv"
path_silva7 = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nh_16S_csv/SILVA_csv/2023-03-07T09_17_52.385Z.csv"


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
columns = ['Organism name', 'Length', 'Benchmark ID', 'Taxonomy.SILVA.superkingdom', 'Taxonomy.SILVA.phylum']

df_results = rebuild_file(path_silva1, columns, df_results)
df_results = rebuild_file(path_silva2, columns, df_results)
df_results = rebuild_file(path_silva3, columns, df_results)
df_results = rebuild_file(path_silva4, columns, df_results)
df_results = rebuild_file(path_silva5, columns, df_results)
df_results = rebuild_file(path_silva6, columns, df_results)
df_results = rebuild_file(path_silva7, columns, df_results)
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