import csv
import pandas as pd
import os
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent
path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_5S/Results/result.xlsx"
path_ENA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/ENA_5S.csv"
path_ENA_destination = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_ENA_5S/"
path_GTDB_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/GTDB_5S.csv"
path_GTDB_destination = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_GTDB_5S/"
path_NCBI_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/NCBI_5S.csv"
path_NCBI_destination = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_NCBI_5S/"
path_SILVA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/SILVA_5S.csv"
path_SILVA_destination = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_SILVA_5S/"

df = pd.read_excel(path)


# Funzione che prende il percorso di un file csv, il path dove andare a creare le subdirectory, la colonna dell'id delle molecole
# e quella del phylum presenti sul file csv. Poi le crea con il nome della riga di riferimento al phylum, se non esiste ancora.
def create_subdirectory(csv_filepath, dest_path, col_id_molecule, col_phylum):
    with open(csv_filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        subdirectory_names = []
        for row in reader:
            if row[col_id_molecule] != '' and row[col_phylum] != '':
                if row[col_id_molecule] in df['benchmark id'].values:
                    # print(row[col_id_molecule], row[col_phylum])
                    if row[col_phylum] not in subdirectory_names:
                        os.mkdir(os.path.join(dest_path, row[col_phylum]))
                        subdirectory_names.append(row[col_phylum])


create_subdirectory(path_ENA_origin, path_ENA_destination,
                    15, 19)

create_subdirectory(path_GTDB_origin, path_GTDB_destination,
                    15, 19)

create_subdirectory(path_NCBI_origin, path_NCBI_destination,
                    15, 21)

create_subdirectory(path_SILVA_origin, path_SILVA_destination,
                    15, 24)
