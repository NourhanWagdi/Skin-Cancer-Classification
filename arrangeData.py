
#%%
import pandas as pd
import os
import util

#%%
df= pd.read_csv(util.METADATA_PATH)
print(df.head())

#%%
distinct_dx_type = df['dx'].unique()
print(distinct_dx_type)

#%%
distinct_img_dir_path = os.path.join(util.SKIN_CANCER_HAMNIST_HAM_1000_PATH, "distinctImages")
os.mkdir(distinct_img_dir_path)
for dx in distinct_dx_type:
    try:
        os.mkdir(os.path.join(distinct_img_dir_path,dx))
    except Exception as e:
        print("failed to make directory.", e)

#%%
imageIds = os.listdir(util.IMAGE_PATH)
#%%
from shutil import copy

for image_id in imageIds:
    temp = image_id.replace(".jpg", "")
    tempDest = df.loc[df['image_id'] == temp, 'dx'].iloc[0]
    dest = os.path.join(os.path.join(distinct_img_dir_path, tempDest), image_id)
    src = os.path.join(util.IMAGE_PATH, image_id)
    copy(src, dest)

#%%
arrangedData = os.path.join(util.SKIN_CANCER_HAMNIST_HAM_1000_PATH, 'arrangedData')
training = os.path.join(arrangedData, 'training')
testing = os.path.join(arrangedData, 'testing')

try:
    #os.mkdir(arrangedData)
    os.mkdir(training)
    os.mkdir(testing)
except Exception as e:
    print(e)

#%%
for dx in distinct_dx_type:
    try:
        os.mkdir(os.path.join(training,dx))
        os.mkdir(os.path.join(testing,dx))
    except Exception as e:
        print("failed to make directory.", e)

#%%
from shutil import copyfile
import random
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = os.path.join(SOURCE, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(TRAINING, filename)
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(TESTING, filename)
        copyfile(this_file, destination)

#%%
splitSize = .9
for dx in distinct_dx_type:
    srcPath = os.path.join(distinct_img_dir_path, dx)
    trainingPath = os.path.join(training, dx)
    testingPath = os.path.join(testing, dx)
    split_data(srcPath, trainingPath, testingPath, splitSize)

#%%
