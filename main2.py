#%%
# import matplotlib.pyplot as plt
import os
import util
import train
import custom_model

TRAINING_DIR = os.path.join(
    util.SKIN_CANCER_HAMNIST_HAM_1000_PATH, 'arrangedData/training')
TESTING_DIR = os.path.join(
    util.SKIN_CANCER_HAMNIST_HAM_1000_PATH, 'arrangedData/testing')
VAL_DIR = os.path.join(
    util.SKIN_CANCER_HAMNIST_HAM_1000_PATH, 'arrangedData/validation')

model = custom_model.get_inception_model(input_shape=(224,224,3))
history = train.trainRaw(model, TRAINING_DIR, TESTING_DIR, VAL_DIR, epochs=50)