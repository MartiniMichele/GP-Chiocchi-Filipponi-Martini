import math
import os
from pathlib import Path
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import numpy as np

# organize data into train, validation and test directories
source_path = Path(__file__).resolve()
source_dir = source_path.parent
data_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/Phylum_SILVA_tRNA/"
dataset_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/"
train_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/train/"
valid_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/valid/"
test_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/test/"

# TODO: eliminare dopo aver spostato i file

filelist = os.listdir(data_dir)
sub_dir_path = []

if len(os.listdir(train_dir)) == 0:
    for x in filelist:
        if x != "DATASET":
            sub_dir_path.append(data_dir + x + '/')
    for x in sub_dir_path:
        print(str(x))

    for i in sub_dir_path:
        filelist2 = os.listdir(i)
        count = 0
        train_part = math.floor(len(filelist2) * 0.7)
        valid_part = math.floor(len(filelist2) * 0.2)
        test_part = math.floor(len(filelist2) * 0.1)
        print("phylum: " + str(i))
        print("70: " + str(train_part))
        print("20: " + str(valid_part))
        print("10: " + str(test_part))

        os.chdir(i)
        for j in random.sample(glob.glob('*.png'), train_part):
            source_path = i + j
            shutil.move(source_path, train_dir)
        for j in random.sample(glob.glob('*.png'), valid_part):
            source_path = i + j
            shutil.move(source_path, valid_dir)
        for j in random.sample(glob.glob('*.png'), test_part):
            source_path = i + j
            shutil.move(source_path, test_dir)


'''
controlla se è presente la cartella "Actinobacteria" comune a tutte e 3 le cartelle(train, valid, test) e quindi sicuramente presente
se si va avanti e crea le sottocartelle(labels) con i nomi dei phylum presenti.

Per creare le cartelle dei labels vengono letti tutti i file .png, viene estratta la prima parte del nome relativa al phylum
e poi ci viene spostato il file
'''
if os.path.isdir(os.path.abspath(os.path.join(train_dir, "Actinobacteria/"))) is False:
    os.chdir(train_dir)
    for file in os.listdir():
        if file.endswith('.png'):
            phylum = file.split('_')[0]
            if os.path.isdir(phylum) is False:
                os.makedirs(phylum)
            dir_path = train_dir + phylum
            shutil.move(file, dir_path)

if os.path.isdir(os.path.abspath(os.path.join(valid_dir, "Actinobacteria/"))) is False:
    os.chdir(valid_dir)
    for file in os.listdir():
        if file.endswith('.png'):
            phylum = file.split('_')[0]
            if os.path.isdir(phylum) is False:
                os.makedirs(phylum)
            dir_path = valid_dir + phylum
            shutil.move(file, dir_path)

if os.path.isdir(os.path.abspath(os.path.join(test_dir, "Actinobacteria/"))) is False:
    os.chdir(test_dir)
    for file in os.listdir():
        if file.endswith('.png'):
            phylum = file.split('_')[0]
            if os.path.isdir(phylum) is False:
                os.makedirs(phylum)
            dir_path = test_dir + phylum
            shutil.move(file, dir_path)
