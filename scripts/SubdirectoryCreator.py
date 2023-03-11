'''
Questo script consente di scorrere una determinata colonna di un file excel e di creare delle cartelle aventi il nome
della riga che al momento si sta considerando.
'''

import csv
import pandas as pd
import os
from pathlib import Path

# Percorsi dei files excel
source_path = Path(__file__).resolve()
source_dir = source_path.parent
path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "//16S_csv/Results_phylum_noduplicates/join_phylum_noduplicates.xlsx"
path_ENA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nH_16S_csv/Results_notnull_superkingdom/ENA.csv"
path_GTDB_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/23S_tassonomie/2023-03-02T13_37_46.490Z.csv"
path_LTP_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/23S_tassonomie/2023-03-02T13_37_22.792Z.csv"
path_NCBI_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/23S_tassonomie/2023-03-02T13_37_35.342Z.csv"
path_SILVA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/23S_tassonomie/2023-03-02T13_36_56.689Z.csv"
path_join_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nH_16S_csv/Results_notnull_phylum/join.csv"

path_join_noduplicates_origin= os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_csv/Results_phylum_noduplicates/join_phylum_noduplicates.csv"
path_join_noduplicates_phylum= os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_16S_join_noduplicates"

path_ENA_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_23S/Phylum_23S_ENA/"
path_GTDB_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_23S/Phylum_23S_GTDB/"
path_LTP_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_23S/Phylum_23S_LTP/"
path_NCBI_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_23S/Phylum_23S_NCBI/"
path_SILVA_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_23S/Phylum_23S_SILVA/"

path_ENA_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_16S_ENA/"
#path_GTDB_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_GTDB_5S/"
#path_NCBI_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_NCBI_5S/"
path_SILVA_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_SILVA_tRNA/"
path_join_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_16S_join/"

path_16S_csv= os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nH_16S/16S.csv"
path_16S_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Superkingdom_16S/"

path_23S_csv= os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nH_23S/23S.csv"
path_23S_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Superkingdom_23S/"

df = pd.read_excel(path)

# Funzione che prende il percorso di un file csv, il path dove andare a creare le subdirectory, la colonna dell'id delle molecole
# e quella del phylum presenti sul file csv. Poi le crea con il nome della riga di riferimento al phylum, se non esiste ancora.
def create_subdirectory(csv_filepath, dest_path, benchmark_id_csv, col_classifier):
    with open(csv_filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        subdirectory_names = []
        for row in reader:
            if row[benchmark_id_csv] != '' and row[col_classifier] != '':
                if row[benchmark_id_csv] in df['Benchmark ID'].values:
                    #print(row[col_id_molecule], row[col_classifier])
                    if row[col_classifier] not in subdirectory_names:
                        os.mkdir(os.path.join(dest_path, row[col_classifier]))
                        subdirectory_names.append(row[col_classifier])


# Crea, se non sono state già create, tutte le cartella in base al nome del classificatore
create_subdirectory(path_join_noduplicates_origin, path_join_noduplicates_phylum, 2, 4)
#create_subdirectory(path_GTDB_origin, path_GTDB_superkingdom, 15, 19)
#create_subdirectory(path_LTP_origin, path_LTP_superkingdom, 15, 19)
#create_subdirectory(path_NCBI_origin, path_NCBI_superkingdom, 15, 21)
#create_subdirectory(path_SILVA_origin, path_SILVA_superkingdom, 15, 24)
#create_subdirectory(path_16S_csv, path_16S_superkingdom, 11, 1)
#create_subdirectory(path_23S_csv, path_23S_superkingdom, 11, 1)

'''
# Crea, se non sono state già create, tutte le cartelle in base al nome del phylum
create_subdirectory(path_ENA_origin, path_ENA_phylum, 15, 19)
create_subdirectory(path_GTDB_origin, path_GTDB_phylum, 15, 19)
create_subdirectory(path_NCBI_origin, path_NCBI_phylum, 15, 21)
create_subdirectory(path_SILVA_origin, path_SILVA_phylum, 15, 24)
'''
