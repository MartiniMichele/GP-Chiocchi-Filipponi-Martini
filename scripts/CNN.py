import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import itertools
import os
from pathlib import Path
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings


# organize data into train, validation and test directories
source_path = Path(__file__).resolve()
source_dir = source_path.parent
data_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Dataset/NCBI"

os.chdir(data_dir)
if os.path.isdir('train/prova') is False:
    os.makedirs('train/prova')
    os.makedirs('train/prova2')
    os.makedirs('valid/prova')
    os.makedirs('valid/prova2')
    os.makedirs('test/prova')
    os.makedirs('test/prova2')
'''
    for i in random.sample(glob.glob('cat*'), 500):
        shutil.move(i, 'train/cat')
    for i in random.sample(glob.glob('dog*'), 500):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 100):
        shutil.move(i, 'valid/cat')
    for i in random.sample(glob.glob('dog*'), 100):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 50):
        shutil.move(i, 'test/cat')
    for i in random.sample(glob.glob('dog*'), 50):
        shutil.move(i, 'test/dog')

os.chdir('../../')
'''





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

