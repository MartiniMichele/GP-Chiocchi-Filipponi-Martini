import shutil
import os
import re

import pandas as pd

df = pd.read_excel('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Fasta_5S/Results/result.xlsx')

image_names = os.listdir('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/UpdatedDataset/')

# Metodo per ordinare le immagini dell'UpdatedDataset.
def sort_key(s):
    return int(re.findall(r'\d+', s)[0])

image_names = [file_name.replace('.png', '') for file_name in image_names]
sorted_image_names = sorted(image_names, key=sort_key)

def copy_image(csv_filepath, dest_path, benchmark_id_csv, col_phylum):
    df_csv = pd.read_csv(csv_filepath)
    count = 0
    for index, row in df.iterrows():
        if row['benchmark id'] in df_csv[benchmark_id_csv].values:
            count = count + 1
            corresponding_row = df_csv.loc[df_csv[benchmark_id_csv] == row['benchmark id']]
            if not pd.isna(corresponding_row[benchmark_id_csv].values[0]) and not pd.isna(corresponding_row[col_phylum].values[0]):
                print(row['benchmark id'], count, corresponding_row[col_phylum].values[0])
                src = os.path.join('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/UpdatedDataset/', f"{sorted_image_names[count-1]}.png")
                dest = os.path.join(dest_path, corresponding_row[col_phylum].values[0], f"{corresponding_row[col_phylum].values[0]}_{count}.png")
                shutil.copy2(src, dest)

copy_image('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/fwd5s/ENA_5S.csv',
              'C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Classificazione/Phylum_ENA_5S/',
              'Benchmark ID', 'Taxonomy.ENA.phylum')

copy_image('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/fwd5s/GTDB_5S.csv',
              'C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Classificazione/Phylum_GTDB_5S/'
           , 'Benchmark ID', 'Taxonomy.GTDB.phylum')

copy_image('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/fwd5s/NCBI_5S.csv',
              'C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Classificazione/Phylum_NCBI_5S/'
           , 'Benchmark ID', 'Taxonomy.NCBI.phylum')

copy_image('C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/fwd5s/SILVA_5S.csv',
              'C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/Classificazione/Phylum_SILVA_5S/'
           , 'Benchmark ID', 'Taxonomy.SILVA.phylum')