import csv

import pandas as pd
import os

df = pd.read_excel('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Fasta_5S/Results/result.xlsx')

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
                    #print(row[col_id_molecule], row[col_phylum])
                    if row[col_phylum] not in subdirectory_names:
                        os.mkdir(os.path.join(dest_path, row[col_phylum]))
                        subdirectory_names.append(row[col_phylum])

create_subdirectory("C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/fwd5s/ENA_5S.csv",
                    "C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Classificazione/Phylum_ENA_5S/", 15, 19)

create_subdirectory('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/fwd5s/GTDB_5S.csv',
                  'C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Classificazione/Phylum_GTDB_5S/', 15, 19)

create_subdirectory('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/fwd5s/NCBI_5S.csv',
                  'C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Classificazione/Phylum_NCBI_5S/', 15, 21)

create_subdirectory('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/fwd5s/SILVA_5S.csv',
                  'C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Classificazione/Phylum_SILVA_5S/', 15, 24)
