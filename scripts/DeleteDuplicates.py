import pandas as pd
import os
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent
path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_5S/Origin_File/5s.xlsx"
path_results = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_5S/Results/result.xlsx"

# Caricamento del file 5s.xlsx
df = pd.read_excel(path)

# Rimozione dei duplicati nella colonna Organism name
df.drop_duplicates(subset='Organism name ', keep='first', inplace=True)

# Elimino i valori nulli della colonna Organism name
df.dropna(subset=['Organism name '], inplace=True)

# Rimuovo le righe con lunghezza inferiore a 100 o maggiore di 130
df = df[(df['Length'] >= 100) & (df['Length'] <= 130)]

# Salvataggio del risultato in un nuovo file xlsx nella cartella results
df.to_excel(path_results, index=False)