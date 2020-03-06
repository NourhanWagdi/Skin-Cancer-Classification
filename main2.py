#%%
# import matplotlib.pyplot as plt
import os
import util
import train
import custom_model
import tensorflow as tf

TRAINING_DIR = os.path.join(
    util.SKIN_CANCER_HAMNIST_HAM_1000_PATH, 'arrangedData/training')
TESTING_DIR = os.path.join(
    util.SKIN_CANCER_HAMNIST_HAM_1000_PATH, 'arrangedData/testing')

model = custom_model.get_inception_model()
history = train.trainRaw(model, TRAINING_DIR, TESTING_DIR, epochs=1)

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# plt.figure()


# plt.show()


#%%
