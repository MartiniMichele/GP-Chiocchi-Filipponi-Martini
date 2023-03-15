'''
Questo script serve per copiare le immagini presenti in un dataset in un determinato percorso, a seconda del nome
del classificatore (in questo caso nome_superkingdom o nome_phylum).
'''

import shutil
import re
import pandas as pd
import os
from pathlib import Path

# Percorsi dei files excel, del dataset e delle subdirectory.
source_path = Path(__file__).resolve()
source_dir = source_path.parent
path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/.16S_ENA/ENA.xlsx"
path_images = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/.16S_ENA/16S_ENA_Dataset/"

path_ENA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/.16S_ENA/ENA.csv"
path_GTDB_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/23S_tassonomie/2023-03-02T13_37_46.490Z.csv"
path_LTP_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/23S_tassonomie/2023-03-02T13_37_22.792Z.csv"
path_NCBI_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/23S_tassonomie/2023-03-02T13_37_35.342Z.csv"
path_SILVA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/23S_tassonomie/2023-03-02T13_36_56.689Z.csv"
path_join_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nH_16S_csv/Results_notnull_phylum/join.csv"
path_ENA_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/.tRNA/Superkingdom/"
path_ENA_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/.16S_ENA/all_phylum/"
path_GTDB_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_23S/Phylum_23S_GTDB/"
path_LTP_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_23S/Phylum_23S_LTP/"
path_NCBI_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_23S/Phylum_23S_NCBI/"
path_SILVA_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_23S/Phylum_23S_SILVA/"
path_16S_csv= os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nH_16S/16S.csv"
path_16S_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Superkingdom_16S/"
path_join_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_16S_join/"
path_23S_csv= os.path.abspath(os.path.join(source_dir, os.pardir)) + "/nH_23S/23S.csv"
path_23S_superkingdom = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Superkingdom_23S/"
path_join_noduplicates_origin= os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/Results/notnull_phylum/join_notnull_noduplicates_phylum.csv"
path_join_noduplicates_phylum= os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/16S_2/Phylum_join_noduplicates/"
path_join2_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/16S_2_csv/Results/notnull_phylum/join_notnull_phylum.csv"
path_join2_phylum = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/16S_2/Phylum_join/"

df = pd.read_excel(path)

image_names = os.listdir(path_images)

# Metodo per ordinare le immagini di un determinato Dataset. La funzione, restituisce il primo numero intero
# presente nella stringa s, permettendo quindi l'ordinamento del tipo -> CGR_RNA1, CGR_RNA2, CGR_RNA3, ...
# invece che l'ordinamento -> CGR_RNA1, CGR_RNA10, CGR_RNA100, ...
def sort_key(s):
    return int(re.findall(r'\d+', s)[0])

# Elimino la stringa '.png' alla fine di ogni file del Dataset
image_names = [file_name.replace('.png', '') for file_name in image_names]
sorted_image_names = sorted(image_names, key=sort_key)

# Copia immagini in un determinato percorso in base al nome del classificatore (superkingdom_class1, ..., phylum_class1, ...)
def copy_image(csv_filepath, dest_path, benchmark_id_csv, col_classificator):
    df_csv = pd.read_csv(csv_filepath)
    count = 0
    for index, row in df.iterrows():
        if row['Benchmark ID'] in df_csv[benchmark_id_csv].values:
            count = count + 1
            corresponding_row = df_csv.loc[df_csv[benchmark_id_csv] == row['Benchmark ID']]
            if not pd.isna(corresponding_row[benchmark_id_csv].values[0]) and not pd.isna(
                    corresponding_row[col_classificator].values[0]):
                print(row['Benchmark ID'], count, corresponding_row[col_classificator].values[0])
                src = os.path.join(path_images, f"{sorted_image_names[count - 1]}.png")
                dest = os.path.join(dest_path, corresponding_row[col_classificator].values[0],
                                    f"{corresponding_row[col_classificator].values[0]}_{count}.png")
                shutil.copy2(src, dest)

# Chiamate per la copia d'immagini
copy_image(path_ENA_origin, path_ENA_phylum, 'Benchmark ID', 'Taxonomy.ENA.phylum')
#copy_image(path_GTDB_origin, path_GTDB_superkingdom, 'Benchmark ID', 'Taxonomy.GTDB.phylum')
#copy_image(path_LTP_origin, path_LTP_superkingdom, 'Benchmark ID', 'Taxonomy.LTP.phylum')
#copy_image(path_NCBI_origin, path_NCBI_superkingdom, 'Benchmark ID', 'Taxonomy.NCBI.phylum')
#copy_image(path_SILVA_origin, path_SILVA_superkingdom, 'Benchmark ID', 'Taxonomy.SILVA.phylum')


'''
copy_image(path_SILVA_origin, path_SILVA_superkingdom, 'Benchmark ID', 'Taxonomy.SILVA.superkingdom')

copy_image(path_ENA_origin, path_ENA_superkingdom
           , 'Benchmark ID', 'Taxonomy.ENA.superkingdom')

copy_image(path_GTDB_origin, path_GTDB_superkingdom
           , 'Benchmark ID', 'Taxonomy.GTDB.domain')

copy_image(path_NCBI_origin, path_NCBI_superkingdom
           , 'Benchmark ID', 'Taxonomy.NCBI.superkingdom')

copy_image(path_SILVA_origin, path_SILVA_superkingdom
           , 'Benchmark ID', 'Taxonomy.SILVA.superkingdom')

copy_image(path_ENA_origin, path_ENA_destination
           , 'Benchmark ID', 'Taxonomy.ENA.phylum')

copy_image(path_GTDB_origin, path_GTDB_destination
           , 'Benchmark ID', 'Taxonomy.GTDB.phylum')

copy_image(path_NCBI_origin, path_NCBI_destination
           , 'Benchmark ID', 'Taxonomy.NCBI.phylum')

copy_image(path_SILVA_origin, path_SILVA_destination
           , 'Benchmark ID', 'Taxonomy.SILVA.phylum')
'''