'''
Questo script consente di contare il numero di occorrenze per un certo dato, contenuto in una colonna di un file excel.
Salva poi il risultato in un altro file.
'''

import pandas as pd

df = pd.read_excel("C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/tRNA_csv/tRNA_csv/Results/NCBI.xlsx")
colonna = df['Taxonomy.NCBI.superkingdom']
occorrenze = {}

for stringa in colonna:
    if not pd.isna(stringa):
        if stringa in occorrenze:
            occorrenze[stringa] += 1
        else:
            occorrenze[stringa] = 1

risultati_df = pd.DataFrame({'Taxonomy.NCBI.superkingdom': list(occorrenze.keys()), 'n° occorrenze': list(occorrenze.values())})

risultati_df.to_excel(
    "C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/tRNA_csv/tRNA_csv/Tassonomie_numero_occorrenze/NCBI_superkingdom_occorrenze.xlsx",
                      index=False)