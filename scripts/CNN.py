from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from pathlib import Path

img_width, img_height = 150, 150
batch_size = 32
epochs = 50
source_path = Path(__file__).resolve()
source_dir = source_path.parent

# cartelle relative al Superkingdom
'''
data_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SK_DATASET/"
train_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SK_DATASET/train/"
valid_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SK_DATASET/valid/"
test_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SK_DATASET/test/"
'''

data_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/"
train_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/train/"
valid_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/valid/"
test_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/test/"

# cartella per il salvataggio del modello della CNN relativo al Phylum
save_dir = os.path.abspath(os.path.join(source_dir,
                                        os.pardir)) + "/CNN_models/SILVA_PHYLUM_model_1_LR1e-4_batch%s_Dropout(0.5)_4layer(32)_epochs(%s).h5" % (
               batch_size, epochs)

# cartella per il salvataggio del modello della CNN relativo al Superkingdom
'''
save_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/CNN_models/SILVA_SUPERKINGDOM_model_1_LR1e-4_batch%s_Dropout(0.5)_4layer(32)_epochs(%s).h5" % (
           batch_size, epochs)
'''


'''
crea i callback per la funzione fit, i callback implementati sono:

early_stopping: ferma prematuramente il training se non ci sono progressi per un numero specifico di epoche(patience)

reduce_lr: riduce il learning rate in caso di appiattimento della curva di apprendimento

model_checkpoint: salva automaticamente il modello con il valore indicato migliore(monitor)
'''
def create_callbacks():
    early_stopping = EarlyStopping(patience=8, monitor='val_loss', verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_lr=0.001,
                                  patience=5, mode='min',
                                  verbose=1)

    model_checkpoint = ModelCheckpoint(monitor='val_loss',
                                       filepath=save_dir,
                                       save_best_only=True,
                                       verbose=1)

    callbacks = [
        early_stopping,
        reduce_lr,
        model_checkpoint
    ]

    return callbacks


'''
il modello della rete neurale
'''
model = Sequential()
model.add(Conv2D(filters=32, activation='relu', kernel_size=(3, 3), input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=512, activation='relu', ))
model.add(Dropout(0.5))
model.add(Dense(units=19, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=1e-4),
              metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

'''
i batch per train/valid/test sono creati con l'ausilio di ImageDataGenerator che permette di applicare
delle modifiche alle immagini. Nel nostro casto per il train viene applicata la data augmentation
'''
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(valid_dir, target_size=(img_width, img_height),
                                                              batch_size=batch_size, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
                                                  batch_size=batch_size, class_mode='categorical')

'''
metodo fit per istruire la rete neurale, i risultati vengono salvati nella variabile history
per eventuali utilizzi futuri(i.e. grafici)
'''
history = model.fit(train_generator, steps_per_epoch=train_generator.n // batch_size, epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n // batch_size,
                    callbacks=create_callbacks())
'''
al termine del training viene chiamato il metodo evaluate sul batch di test per verificare il training della rete
vengono stampate le metriche e viene calcolata la metrica F1
'''
score = model.evaluate(test_generator, verbose=2)
print('\nTest LOSS: ', score[0])
print('\nTest ACCURACY: ', score[1])
print('\nTest PRECISION: ', score[2])
print('\nTest RECALL: ', score[3])
print('\nTest AUC: ', score[4])
print('\nTest F1: ', str(2 * (score[3] * score[2]) / (score[3] + score[2])))
