import os
import matplotlib.pyplot as plt
from pathlib import Path
import keras.models
from keras_preprocessing.image import ImageDataGenerator

img_width, img_height = 150, 150
database = "16S_JOIN"
superkingdom = "PHYLUM"
livello = "%s_%s" % (database, superkingdom)
model_mk = 1
batch_size = 32
epochs = 20
fl_filter = 32
ol_units = 4
n_dropout = 1
n_layer = 3
lr = 1e-4

source_path = Path(__file__).resolve()
source_dir = source_path.parent
test_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/Classification/%s_DATASET/test/" % database
save_model_dir = os.path.abspath(os.path.join(source_dir, os.pardir)) + "/CNN_models/%s/" % livello


model_filename = "%s_model_%s_LR%s_batch%s_%sDropout(0.5)_%slayer(FL=%s)_epochs(%s)" % (
    livello,
    model_mk, lr,
    batch_size,
    n_dropout,
    n_layer,
    fl_filter,
    epochs)

save_dir = os.path.abspath(os.path.join(save_model_dir, model_filename + ".h5"))

model = keras.models.load_model(save_dir)
model.summary()

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
                                                  batch_size=batch_size, class_mode='categorical')


score = model.evaluate(test_generator, verbose=2)

test_loss = score[0]
test_accuracy = score[1]
test_precision = score[2]
test_recall = score[3]
test_auc = score[4]
test_f1 = 2 * (score[3] * score[2]) / (score[3] + score[2])

print('\nTest LOSS: ', str(test_loss))
print('\nTest ACCURACY: ', str(test_accuracy))
print('\nTest PRECISION: ', str(test_precision))
print('\nTest RECALL: ', str(test_recall))
print('\nTest AUC: ', str(test_auc))
print('\nTest F1: ', str(test_f1))
