import shutil
import os
import re
import pandas as pd
import os
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent
path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Fasta_5S/Results/result.xlsx"
path_images = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/UpdatedDataset/"
path_ENA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/ENA_5S.csv"
path_ENA_destination = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_ENA_5S/"
path_GTDB_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/GTDB_5S.csv"
path_GTDB_destination = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_GTDB_5S/"
path_NCBI_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/NCBI_5S.csv"
path_NCBI_destination = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_NCBI_5S/"
path_SILVA_origin = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/fwd5s/SILVA_5S.csv"
path_SILVA_destination = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_SILVA_5S/"

df = pd.read_excel(path)

image_names = os.listdir(path_images)


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
            if not pd.isna(corresponding_row[benchmark_id_csv].values[0]) and not pd.isna(
                    corresponding_row[col_phylum].values[0]):
                print(row['benchmark id'], count, corresponding_row[col_phylum].values[0])
                src = os.path.join(path_images, f"{sorted_image_names[count - 1]}.png")
                dest = os.path.join(dest_path, corresponding_row[col_phylum].values[0],
                                    f"{corresponding_row[col_phylum].values[0]}_{count}.png")
                shutil.copy2(src, dest)


copy_image(path_ENA_origin, path_ENA_destination
           , 'Benchmark ID', 'Taxonomy.ENA.phylum')

copy_image(path_GTDB_origin, path_GTDB_destination
           , 'Benchmark ID', 'Taxonomy.GTDB.phylum')

copy_image(path_NCBI_origin, path_NCBI_destination
           , 'Benchmark ID', 'Taxonomy.NCBI.phylum')

copy_image(path_SILVA_origin, path_SILVA_destination
           , 'Benchmark ID', 'Taxonomy.SILVA.phylum')
