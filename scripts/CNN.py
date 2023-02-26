import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, History
import os
from pathlib import Path

source_path = Path(__file__).resolve()
source_dir = source_path.parent
img_width, img_height = 150, 150
batch_size = 8
epochs = 50

# variabili di comodo per salvataggio modello
database = "SILVA"
livello = "SPLIT_PHYLUM"
model_mk = 1
fl_filter = 32
n_dropout = 1
lr = 1e-4

# cartelle relative al Superkingdom
'''
data_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SK_DATASET/"
train_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SK_DATASET/train/"
valid_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SK_DATASET/valid/"
test_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SK_DATASET/test/"
'''

# cartelle relative al Phylum

'''
data_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/"
train_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/train/"
valid_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/valid/"
test_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/DATASET/test/"
'''

data_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SPLIT_DATASET/"
train_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SPLIT_DATASET/Archaea/train/"
valid_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SPLIT_DATASET/Archaea/valid/"
test_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/SPLIT_DATASET/Archaea/test/"
save_fig_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Grafici/train_validation/"

model_filename = "%s_%s_model_%s_LR%s_batch%s_%sDropout(0.5)_4layer(FL=%s)_epochs(%s)" % (
    database, livello,
    model_mk, lr,
    batch_size,
    n_dropout,
    fl_filter,
    epochs)

save_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/CNN_models/" + model_filename + ".h5"

print(save_dir)

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
    hist = History()

    callbacks = [
        early_stopping,
        reduce_lr,
        model_checkpoint,
        hist
    ]

    return callbacks


'''
il modello della rete neurale
'''
model = Sequential()
model.add(Conv2D(filters=fl_filter, activation='relu', kernel_size=(3, 3), input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=fl_filter * 2, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=fl_filter * 4, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=fl_filter * 16, activation='relu', ))
model.add(Dropout(0.5))
model.add(Dense(units=2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=lr),
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

actual_epochs = len(history.history['loss'])
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_precision = history.history['precision']
val_precision = history.history['val_precision']
train_recall = history.history['recall']
val_recall = history.history['val_recall']
train_auc = history.history['auc']
val_auc = history.history['val_auc']
# train_F1 = 2*(history.history['recall'] * history.history['precision']) / (history.history['recall'] + history.history['precision'])
# val_F1 = 2*(history.history['val_recall'] * history.history['val_precision']) / (history.history['val_recall'] + history.history['val_precision'])
xc = range(actual_epochs)



if os.path.isdir(os.path.abspath(os.path.join(save_fig_dir, "GRAPHS_" + model_filename))) is False:
    os.chdir(save_fig_dir)
    os.makedirs("GRAPHS_" + model_filename)
    print("CARTELLA CREATA")

save_fig_dir = os.path.abspath(os.path.abspath(os.path.join(save_fig_dir, "GRAPHS_" + model_filename)))
os.chdir(save_fig_dir)



plt.figure()
plt.grid()
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel("epoche")
plt.ylabel("loss")
plt.title("LOSS GRAPH")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("LOSS_GRAPH.png")
plt.show()

plt.grid()
plt.plot(train_acc)
plt.plot(val_acc)
plt.xlabel("epoche")
plt.ylabel("accuracy")
plt.title("ACCURACY GRAPH")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("ACCURACY_GRAPH.png")
plt.show()

plt.grid()
plt.plot(train_precision)
plt.plot(val_precision)
plt.xlabel("epoche")
plt.ylabel("precision")
plt.title("PRECISION GRAPH")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("PRECISION_GRAPH.png")
plt.show()

plt.grid()
plt.plot(train_recall)
plt.plot(val_recall)
plt.xlabel("epoche")
plt.ylabel("recall")
plt.title("RECALL GRAPH")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("RECALL_GRAPH.png")
plt.show()

plt.grid()
plt.plot(train_auc)
plt.plot(val_auc)
plt.xlabel("epoche")
plt.ylabel("auc")
plt.title("AUC GRAPH")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("AUC_GRAPH.png")
plt.show()

'''
plt.plot(train_F1)
plt.plot(val_F1)
plt.xlabel("epoche")
plt.ylabel("F1")
plt.title("F1 GRAPH")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("F1_GRAPH.png")
plt.show()
'''

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


test_f1 = 2 * (score[3] * score[2]) / (score[3] + score[2])

plt.grid()
plt.scatter([1], test_f1, zorder=5)
plt.ylabel("F1 metric")
plt.title("F1 GRAPH")
plt.savefig("F1(test)_GRAPH.png")
plt.show()
