import shutil

import pandas as pd
import os

# Caricamento del file result.xlsx
df = pd.read_excel('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Fasta_5S/Results/result.xlsx')

# Itero attraverso tutti i file fasta nella cartella Fasta_5S
fasta_folder = "C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Fasta_5S/"
for filename in os.listdir(fasta_folder):
    if filename.endswith(".fasta"):
        file_path = os.path.join(fasta_folder, filename)

    # Apro il file fasta ed estraggo la stringa iniziale
    with open(file_path, "r") as f:
        fasta_string = f.readline().strip()
        if "|" in fasta_string:
            target_string = fasta_string.split(">")[1].split("|")[0]
            # Verifico se la stringa è presente nella colonna del file result.xlsl
            if target_string in df["bpRNA ID"].values:
                # Copio il file fasta nella directory New_Directory
                new_folder = "C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Fasta_5S/New_Directory/"
                new_file_path = os.path.join(new_folder, filename)
                shutil.copy2(file_path, new_file_path)
        else:
            target_string = fasta_string.split(">")[1].split("\n")[0]
            # Verifico se la stringa è presente nella colonna del file result.xlsl
            if target_string in df["Reference Name"].values:
                # Copio il file fasta nella directory New_Directory
                new_folder = "C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Fasta_5S/New_Directory/"
                new_file_path = os.path.join(new_folder, filename)
                shutil.copy2(file_path, new_file_path)



