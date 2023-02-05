import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
from pathlib import Path
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# organize data into train, validation and test directories
source_path = Path(__file__).resolve()
source_dir = source_path.parent
data_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_ENA_5S/Only_PNG/"
train_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_ENA_5S/Only_PNG/train/"
valid_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_ENA_5S/Only_PNG/valid/"
test_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_ENA_5S/Only_PNG/test/"

'''
# TODO: eliminare dopo aver spostato i file
png_path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_ENA_5S/"
filelist = os.listdir(png_path)
sub_dir_path = []

for x in filelist:
    sub_dir_path.append(png_path + x + '/')
print(sub_dir_path)

for i in sub_dir_path:
    filelist2 = os.listdir(i)
    for j in filelist2:
        if j.endswith('.png'):
            # file = pd.read_csv(i+j, sep = ',', header = [0])
            file_path = i + j
            new_file_path = png_path
            shutil.copy2(file_path, new_file_path)
'''

os.chdir(data_dir)
if os.path.isdir('train/') is False:
    os.makedirs('train/')
    os.makedirs('valid/')
    os.makedirs('test/')

    for i in random.sample(glob.glob('*.png'), 427):
        shutil.move(i, train_dir)
    for i in random.sample(glob.glob('*.png'), 122):
        shutil.move(i, valid_dir)
    for i in random.sample(glob.glob('*.png'), 61):
        shutil.move(i, test_dir)

# TODO: decidere se lasciare o eliminare (il metodo crea dinamicamente le cartelle per i file presenti nella cartella)
png_path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classificazione/Phylum_ENA_5S/Only_PNG/train"
os.chdir(png_path)
phylum_list = set()

for file in os.listdir():
    if file.endswith('.png'):
        phylum = file.split('_')[0]
        phylum_list.add(str(phylum))
    phylum_list.add(file)

#for item in phylum_list:
#    os.makedirs(item)

for file in os.listdir():
    if file.endswith('.png'):
        filename = file.split('_')[0]
        dir_path = png_path + "/" + filename
        shutil.move(file, dir_path)

os.chdir('../../')
print("phylums:---------------------" + str(phylum_list))
print("phylums lenght:---------------------" + str(len(phylum_list)))

'''
phylums = ['Acidobacteria', 'Actinobacteria', 'Annelida', 'Apicomplexa', 'Aquificae', 'Arthropoda', 'Ascomycota',
          'Bacteroidetes', 'Basidiomycota', 'Brachiopoda', 'Bryozoa', 'Chlamydiae', 'Chlorobi', 'Chloroflexi',
          'Chlorophyta', 'Chordata', 'Ciliophora', 'Cnidaria', 'Crenarchaeota', 'Cryptophyta', 'Cyanobacteria',
          'Deinococcus-Thermus', 'Echinodermata', 'Euglenida', 'Euryarchaeota', 'Firmicutes', 'Hemichordata',
          'Mollusca', 'Mucoromycota', 'Nematoda', 'Nemertea', 'Placozoa', 'Placozoa', 'Platyhelminthes', 'Porifera',
          'Proteobacteria', 'Rhodophyta', 'Rotifera', 'Spirochaetes', 'Streptophyta', 'Tenericutes', 'Thermotogae',
          'Verrucomicrobia', 'Zoopagomycota']
'''

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_dir, target_size=(600, 600),
                         classes=phylum_list, batch_size=2)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_dir, target_size=(600, 600),
                         classes=phylum_list, batch_size=2)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_dir, target_size=(600, 600), classes=phylum_list,
                         batch_size=2, shuffle=False)

imgs, labels = next(train_batches)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plotImages(imgs)
print(labels)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(600, 600, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(units=len(phylum_list), activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches,
          steps_per_epoch=len(train_batches) / 2,
          validation_data=valid_batches,
          validation_steps=len(valid_batches) / 2,
          epochs=20,
          verbose=2
          )

'''
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from PIL import Image

# Load my Dataset
x = []
y = []
data_folder = "C:/Users/Michele/Documents/GitHub/GP-Chiocchi-Filipponi-Martini/Dataset"
classes = os.listdir(data_folder)
for c in range(len(classes)):
    class_folder = os.path.join(data_folder, classes[c])
    print("class_folder = ", class_folder)
    for img_file in os.listdir(data_folder):
        img = Image.open(os.path.join(data_folder, img_file))
        img = img.resize((128, 128))
        x.append(np.array(img))
        y.append(c)
x = np.array(x)
y = np.array(y)


# Split data into training, validation, and test sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, len(classes))
y_val = keras.utils.to_categorical(y_val, len(classes))
y_test = keras.utils.to_categorical(y_test, len(classes))

# Create the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=x.shape[1:]))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(len(classes), activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)
'''
