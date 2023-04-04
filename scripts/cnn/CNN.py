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
source_dir = Path(source_path.parent.parent.parent)
img_width, img_height = 150, 150

'''
ISTRUZIONI PER L'USO:
- inserire il nome del database da utilizzare !!!Attenzione agli "_"!!! (NEW_16S_, NEW_tRNA_, 23S, SK_DATASET....)
- inserire il livello, in caso non fosse presente lasciare vuoto (BACTERIA, ALLPHYLUM, SUPERKINGDOM....)
- scegliere gli iperparametri per la rete
- N.B. il modello viene salvato in automatico nella cartella CNN_models, questo file può pesare molto.
  Ricordarsi di eliminarlo se non è utile
- alla fine del processo di training vengono generati dei grafici che saranno salvati nella cartella Grafici
- una volta finito il training sara effettuata una evaluation sul test e saranno visualizzati i risultati sulla console.
  Verranno inoltre generati dei grafici che saranno salvati nella suddetta cartella
'''
# variabili di comodo per creazione e salvataggio del modello
database = "NEW_16S_"
livello = "ALLPHYLUM"
completo = "%s%s" % (database, livello)
#variabile di comodo in caso si vogliano lasciare gli stessi parametri ma ripetere il training senza riscrivere il modello salvato
model_mk = 1
batch_size = 32
epochs = 30
#numero di filtri del primo layer
fl_filter = 32
#numero di unità del layer di output
ol_units = 12
#numero di layer di dropout(0.5)
n_dropout = 1
#numero di layer totali della rete
n_layer = 3
#learning rate
lr = 1e-4

# cartelle urilizzate per l'esperimento
data_dir = Path(str(source_dir) + "/Classification/%s_DATASET/" % completo)
train_dir = Path(str(source_dir) + "/Classification/%s_DATASET/train/" % completo)
valid_dir = Path(str(source_dir) + "/Classification/%s_DATASET/valid/" % completo)
test_dir = Path(str(source_dir) + "/Classification/%s_DATASET/test/" % completo)

models_dir = Path(str(source_dir) + "/CNN_models/")
save_model_dir = Path(str(source_dir) + "/CNN_models/%s/" % livello)
graph_dir = Path(str(source_dir) + "/Grafici/")
save_fig_dir = Path(str(source_dir) + "/Grafici/%s" % livello)

model_filename = "%s_model_%s_LR%s_batch%s_%sDropout(0.5)_%slayer(FL=%s)_epochs(%s)" % (
    livello,
    model_mk, lr,
    batch_size,
    n_dropout,
    n_layer,
    fl_filter,
    epochs)

if os.path.isdir(save_model_dir) is False:
    os.chdir(source_dir)
    os.makedirs(save_model_dir)
    print("\nCARTELLA SALVATAGGIO MODELLO CREATA")

model_save_name = os.path.abspath(os.path.join(save_model_dir, model_filename + ".h5"))

print("\n" + model_save_name)

'''
crea i callback per la funzione fit, i callback implementati sono:
early_stopping: ferma prematuramente il training se non ci sono progressi per un numero specifico di epoche(patience)
reduce_lr: riduce il learning rate in caso di appiattimento della curva di apprendimento
model_checkpoint: salva automaticamente il modello con il valore indicato migliore(monitor)
'''


def create_callbacks():
    early_stopping = EarlyStopping(patience=6, monitor='val_loss', verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_lr=0.001,
                                  patience=5, mode='min',
                                  verbose=1)

    model_checkpoint = ModelCheckpoint(monitor='val_loss',
                                       filepath=model_save_name,
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
last_n_filter = fl_filter
model = Sequential()
for i in range(n_layer):
    if i == 0:
        model.add(Conv2D(filters=last_n_filter, activation='relu', kernel_size=(3, 3), input_shape=(img_width, img_height, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    else:
        last_n_filter = last_n_filter * 2
        model.add(Conv2D(filters=last_n_filter, activation='relu', kernel_size=(3, 3), input_shape=(img_width, img_height, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=last_n_filter * 4, activation='relu', ))

for i in range(n_dropout):
    model.add(Dropout(0.5))

model.add(Dense(units=ol_units, activation='softmax'))


'''
MODELLO FUNZIONANTE:


model.add(Conv2D(filters=32, activation='relu', kernel_size=(3, 3), input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=512, activation='relu', ))
model.add(Dropout(0.5))
model.add(Dense(units=8, activation='softmax'))
'''

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

'''
variabili per la creazione dei grafici, estratte dalla storia del metodo fit
'''
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
xc = range(actual_epochs)


'''
Controlla se la cartella del superkingdom del modello esiste e se necessario la crea
'''
if os.path.isdir(save_fig_dir) is False:
    os.chdir(graph_dir)
    os.makedirs(livello)
    print("CARTELLA SUPERKINGDOM SALVATAGGIO GRAFICI CREATA")
'''
Controlla se la cartella dei grafici del modello esiste e se necessario la crea
'''
if os.path.isdir(os.path.abspath(os.path.join(save_fig_dir, "GRAPHS_" + model_filename))) is False:
    os.chdir(save_fig_dir)
    os.makedirs("GRAPHS_" + model_filename)
    print("CARTELLA SALVATAGGIO GRAFICI CREATA")

'''
aggiorna la variabile del salvataggio dei grafici e si sposta in quella cartella
'''
save_fig_dir = os.path.abspath(os.path.abspath(os.path.join(save_fig_dir, "GRAPHS_" + model_filename)))
os.chdir(save_fig_dir)


'''
Crea il grafico della loss per train e validation
'''
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

'''
Crea il grafico della accuracy per train e validation
'''
plt.grid()
plt.plot(train_acc)
plt.plot(val_acc)
plt.xlabel("epoche")
plt.ylabel("accuracy")
plt.title("ACCURACY GRAPH")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("ACCURACY_GRAPH.png")
plt.show()

'''
Crea il grafico della precision per train e validation
'''
plt.grid()
plt.plot(train_precision)
plt.plot(val_precision)
plt.xlabel("epoche")
plt.ylabel("precision")
plt.title("PRECISION GRAPH")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("PRECISION_GRAPH.png")
plt.show()

'''
Crea il grafico del recall per train e validation
'''
plt.grid()
plt.plot(train_recall)
plt.plot(val_recall)
plt.xlabel("epoche")
plt.ylabel("recall")
plt.title("RECALL GRAPH")
plt.legend(["train", "val"], loc="upper left")
plt.savefig("RECALL_GRAPH.png")
plt.show()

'''
Crea il grafico dell'AUC per train e validation
'''
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
al termine del training viene chiamato il metodo evaluate sul batch di test per verificare il training della rete
vengono stampate le metriche e viene calcolata la metrica F1
'''

score = model.evaluate(test_generator, verbose=2)
test_loss = score[0]
test_accuracy = score[1]
test_precision = score[2]
test_recall = score[3]
test_auc = score[4]

print('\nRISULTATO TEST %s (BATCH: %s, EPOCHE: %s, STRATI: %s, FIRST LAYER: %s, ) %s' % (model_mk, batch_size, actual_epochs, n_layer, fl_filter, completo))
print('\nTest LOSS: ', str(test_loss))
print('\nTest ACCURACY: ', str(test_accuracy))
print('\nTest PRECISION: ', str(test_precision))
print('\nTest RECALL: ', str(test_recall))
print('\nTest AUC: ', str(test_auc))


'''
Crea il grafico della loss per il test
'''
plt.grid()
plt.scatter([1], test_loss, zorder=5)
plt.ylabel("loss")
plt.title("LOSS GRAPH")
plt.savefig("LOSS(test)_GRAPH.png")
plt.show()

'''
Crea il grafico della accuracy per il test
'''
plt.grid()
plt.scatter([1], test_accuracy, zorder=5)
plt.ylabel("accuracy")
plt.title("ACCURACY GRAPH")
plt.savefig("ACCURACY(test)_GRAPH.png")
plt.show()

'''
Crea il grafico della precision per il test
'''
plt.grid()
plt.scatter([1], test_precision, zorder=5)
plt.ylabel("precision")
plt.title("PRECISION GRAPH")
plt.savefig("PRECISION(test)_GRAPH.png")
plt.show()

'''
Crea il grafico del recall per il test
'''
plt.grid()
plt.scatter([1], test_recall, zorder=5)
plt.ylabel("recall")
plt.title("RECALL GRAPH")
plt.savefig("RECALL(test)_GRAPH.png")
plt.show()

'''
Crea il grafico dell'AUC per il test
'''
plt.grid()
plt.scatter([1], test_auc, zorder=5)
plt.ylabel("auc metric")
plt.title("AUC GRAPH")
plt.savefig("AUC(test)_GRAPH.png")
plt.show()


test_f1 = 2 * (score[3] * score[2]) / (score[3] + score[2])
print('\nTest F1: ', str(test_f1))
'''
Crea il grafico dell'F1 per il test
'''
plt.grid()
plt.scatter([1], test_f1, zorder=5)
plt.ylabel("F1 metric")
plt.title("F1 GRAPH")
plt.savefig("F1(test)_GRAPH.png")
plt.show()


