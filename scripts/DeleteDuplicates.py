import pandas as pd

# Caricamento del file 5s.xlsx
df = pd.read_excel('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Fasta_5S/Origin_File/5s.xlsx')

# Rimozione dei duplicati nella colonna Organism name
df.drop_duplicates(subset='Organism name ', keep='first', inplace=True)

# Elimino i valori nulli della colonna Organism name
df.dropna(subset=['Organism name '], inplace=True)

# Rimuovo le righe con lunghezza inferiore a 100 o maggiore di 130
df = df[(df['Length'] >= 100) & (df['Length'] <= 130)]

# Salvataggio del risultato in un nuovo file xlsx nella cartella results
df.to_excel('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Fasta_5S/Results/result.xlsx', index=False)