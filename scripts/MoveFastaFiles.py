import shutil
import pandas as pd
import os
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent
path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/tRNA_csv/tRNA_csv/join2.xlsx"
path_fasta_folder = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_tRNA_nH/Fasta_tRNA_nH/"
path_new_fasta_folder = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_tRNA_nH/Fasta_updated/"

df = pd.read_excel(path)

benchmark_id_xlsx = 'Benchmark ID'
molecule_id = set(df[benchmark_id_xlsx] + '_nH')

for file_fasta in os.listdir(path_fasta_folder):
    filename = os.path.splitext(file_fasta)[0].replace('_nH.fasta', '')
    if filename in molecule_id:
        if not os.path.exists(path_new_fasta_folder):
            os.makedirs(path_new_fasta_folder)
        shutil.copy2(os.path.join(path_fasta_folder, file_fasta), os.path.join(path_new_fasta_folder, file_fasta))



''''
# Itero attraverso tutti i file fasta nella cartella Fasta_tRNA_nH
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

'''
