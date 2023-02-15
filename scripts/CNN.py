from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 150, 150
train_data_dir = 'C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/cnn_files/Superkingdom_ENA_5S/dataset/train'
validation_data_dir = 'C:/Users/fchio/Desktop/GroupProject/GP-Chiocchi-Filipponi-Martini/cnn_files/Superkingdom_ENA_5S/dataset/test'
batch_size = 32
epochs = 30

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.n // batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.n // batch_size)

score = model.evaluate_generator(validation_generator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



'''import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from keras import regularizers, layers
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
data_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/cnn_files/Superkingdom_ENA_5S/dataset/"
train_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/cnn_files/Superkingdom_ENA_5S/dataset/train/"
valid_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/cnn_files/Superkingdom_ENA_5S/dataset/valid/"
test_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/cnn_files/Superkingdom_ENA_5S/dataset/test/"

# TODO: eliminare dopo aver spostato i file

png_path = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/cnn_files/Superkingdom_ENA_5S/"
filelist = os.listdir(png_path)
sub_dir_path = []

for x in filelist:
    sub_dir_path.append(png_path + x + '/')
print("subdir:" + str(sub_dir_path))

for i in sub_dir_path:
    filelist2 = os.listdir(i)
    for j in filelist2:
        if j.endswith('.png'):
            #file = pd.read_csv(i+j, sep = ',', header = [0])
            file_path = i + j
            new_file_path = png_path
            shutil.copy2(file_path, new_file_path)



os.chdir(data_dir)
if os.path.isdir('train/') is False:
    os.makedirs('train/')
    os.makedirs('valid/')
    os.makedirs('test/')

    for i in random.sample(glob.glob('*.png'), 448):
        shutil.move(i, train_dir)
    for i in random.sample(glob.glob('*.png'), 127):
        shutil.move(i, valid_dir)
    for i in random.sample(glob.glob('*.png'), 63):
        shutil.move(i, test_dir)

# TODO: decidere se lasciare o eliminare (il metodo crea dinamicamente le cartelle per i file presenti nella cartella)
os.chdir(train_dir)
phylum_list = set()

for file in os.listdir():
    if file.endswith('.png'):
        phylum = file.split('_')[0]
        phylum_list.add(str(phylum))
    phylum_list.add(file)

if os.path.isdir('Archaea/') is False:
    for item in phylum_list:
        os.makedirs(item)

for file in os.listdir():
    if file.endswith('.png'):
        filename = file.split('_')[0]
        dir_path = train_dir + filename
        shutil.move(file, dir_path)

print("phylums:---------------------" + str(phylum_list))
print("phylums lenght:---------------------" + str(len(phylum_list)))


train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_dir, target_size=(600, 600),
                         classes=phylum_list, batch_size=20)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_dir, target_size=(600, 600),
                         classes=phylum_list, batch_size=20)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_dir, target_size=(600, 600), classes=phylum_list,
                         batch_size=20, shuffle=False)

imgs, labels = next(train_batches)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(imgs)
#print(labels)

# TODO: DATA AUGMENTATION

gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15,
                         zoom_range=0.1, channel_shift_range=10., horizontal_flip=True)

for file in os.listdir():
    if file.endswith('.png'):
        pass
    png_path = train_dir + file
    chosen_image_path = random.choice(os.listdir(png_path))
    image = np.expand_dims(plt.imread(os.path.join(png_path, chosen_image_path)), 0)
    plt.imshow(image[0])
    aug_iter = gen.flow(image)
    aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
    plotImages(aug_images)
    gen.flow(image, save_to_dir=png_path, save_prefix='augimage-', save_format='png')


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])
resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(600, 600),
  layers.Rescaling(1./255)
])

model = Sequential([
    #resize_and_rescale,
    #data_augmentation,
    Conv2D(filters=16, kernel_regularizer=regularizers.L2(0.001), kernel_size=(3, 3), activation='relu', padding='same', input_shape=(600, 600, 3)),
    Dropout(0.5),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Dropout(0.5),
    Dropout(0.5),
    Flatten(),
    Dense(units=len(phylum_list), activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])

print(str(train_batches.samples))
print(str(train_batches.batch_size))
print(str(int(round(train_batches.samples / train_batches.batch_size))))

model.fit(x=train_batches,
          steps_per_epoch=int(round(train_batches.samples / train_batches.batch_size)),
          validation_data=valid_batches,
          validation_steps=int(round(valid_batches.samples / valid_batches.batch_size)),
          epochs=50,
          verbose=1
          )
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
