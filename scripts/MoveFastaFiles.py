import shutil
import pandas as pd
import os
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent
path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/risultato.xlsx"

# Caricamento del file result.xlsx
df = pd.read_excel(path)

path_fasta_folder = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_5S/"
# Itero attraverso tutti i file fasta nella cartella Fasta_5S
for filename in os.listdir(path_fasta_folder):
    if filename.endswith(".fasta"):
        file_path = os.path.join(path_fasta_folder, filename)

    # Apro il file fasta ed estraggo la stringa iniziale
    with open(file_path, "r") as f:
        fasta_string = f.readline().strip()
        # Avendo due database (CRW e RFAM) con rappresentazione sintattica dei file fasta diversi controllo se la stringa "|"
        # è contenuta nel file. Se è contenuta allora passo alla verifica della presenza nella colonna di result.xlsl e lo copio
        # in una nuova cartella;
        if "|" in fasta_string:
            target_string = fasta_string.split(">")[1].split("|")[0]
            if target_string in df["bpRNA ID"].values:
                new_directory = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_5S/New_Directory2/"
                new_file_path = os.path.join(new_directory, filename)
                shutil.copy2(file_path, new_file_path)
        # altrimenti è un file fasta del database RFAM, quindi prendo la stringa, verifico se la stringa è presente nella colonna
        # del file result.xlsl e se si allora la copio nella nuova cartella.
        else:
            target_string = fasta_string.split(">")[1].split("\n")[0]
            if target_string in df["Reference Name"].values:
                new_directory = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_5S/New_Directory2/"
                new_file_path = os.path.join(new_directory, filename)
                shutil.copy2(file_path, new_file_path)


