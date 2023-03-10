'''
Questo script serve per solamente per eliminare righe, il cui contenuto, filtrato sulla colonna, è vuoto.
Viene poi salvato il risultato su un nuovo file excel/csv, a seconda dell'occorrenza.
'''

import pandas as pd
import os
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent
path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/Results/NCBI.xlsx"
path_results = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/Results/notnull_superkingdom/NCBI.csv"

df = pd.read_excel(path)

df.dropna(subset=['Taxonomy.NCBI.superkingdom'], inplace=True)

# Salvataggio del risultato in un nuovo file xlsx nella cartella results
#df.to_excel(path_results, index=False)

df.to_csv(path_results, index=False)