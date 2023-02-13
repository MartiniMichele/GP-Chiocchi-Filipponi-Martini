import csv
import pandas as pd
import os
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent
path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_5S/Results/result.xlsx"
path_ENA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/ENA_5S.csv"
path_GTDB_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/GTDB_5S.csv"
path_NCBI_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/NCBI_5S.csv"
path_SILVA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/SILVA_5S.csv"

path_ENA_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Superkingdom_ENA_5S/"
path_GTDB_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Superkingdom_GTDB_5S/"
path_NCBI_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Superkingdom_NCBI_5S/"
path_SILVA_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Superkingdom_SILVA_5S/"

path_ENA_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_ENA_5S/"
path_GTDB_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_GTDB_5S/"
path_NCBI_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_NCBI_5S/"
path_SILVA_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_SILVA_5S/"

df = pd.read_excel(path)

# Funzione che prende il percorso di un file csv, il path dove andare a creare le subdirectory, la colonna dell'id delle molecole
# e quella del phylum presenti sul file csv. Poi le crea con il nome della riga di riferimento al phylum, se non esiste ancora.
def create_subdirectory(csv_filepath, dest_path, col_id_molecule, col_classifier):
    with open(csv_filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        subdirectory_names = []
        for row in reader:
            if row[col_id_molecule] != '' and row[col_classifier] != '':
                if row[col_id_molecule] in df['benchmark id'].values:
                    # print(row[col_id_molecule], row[col_phylum])
                    if row[col_classifier] not in subdirectory_names:
                        os.mkdir(os.path.join(dest_path, row[col_classifier]))
                        subdirectory_names.append(row[col_classifier])


# Crea, se non sono state già create, tutte le cartella in base al nome del superkingdom
create_subdirectory(path_ENA_origin, path_ENA_superkingdom, 15, 18)
create_subdirectory(path_GTDB_origin, path_GTDB_superkingdom, 15, 18)
create_subdirectory(path_NCBI_origin, path_NCBI_superkingdom, 15, 20)
create_subdirectory(path_SILVA_origin, path_SILVA_superkingdom, 15, 18)

# Crea, se non sono state già create, tutte le cartelle in base al nome del phylum
create_subdirectory(path_ENA_origin, path_ENA_phylum, 15, 19)
create_subdirectory(path_GTDB_origin, path_GTDB_phylum, 15, 19)
create_subdirectory(path_NCBI_origin, path_NCBI_phylum, 15, 21)
create_subdirectory(path_SILVA_origin, path_SILVA_phylum, 15, 24)

