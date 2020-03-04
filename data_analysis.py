# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import util

from keras.preprocessing import image


#%%
df = pd.read_csv(util.METADATA_PATH)
print(df.head())

#%%
image_dir = util.IMAGE_PATH
target_size=(224, 224, 3)
print("Reading raw data...")
images = []
for image_path in os.listdir(image_dir):
    img = image.load_img(os.path.join(
        image_dir, image_path), target_size=target_size)
    images.append(image.img_to_array(img))
X_orig = np.asarray(images)
# Free up memory from images
del images

#%%
fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
df['dx'].value_counts().plot(kind='bar', ax=ax1)


#%%
# this will tell us how many images are associated with each lesion_id
df_undup = df.groupby('lesion_id').count()
# now we filter out lesion_id's that have only one image associated with it
df_undup = df_undup[df_undup['image_id'] == 1]
df_undup.reset_index(inplace=True)
df_undup.head()


# here we identify lesion_id's that have duplicate images and those that have only one image.
def get_duplicates(x):
    unique_list = list(df_undup['lesion_id'])
    if x in unique_list:
        return 'unduplicated'
    else:
        return 'duplicated'

# create a new colum that is a copy of the lesion_id column
df['duplicates'] = df['lesion_id']
# apply the function to this new column
df['duplicates'] = df['duplicates'].apply(get_duplicates)
df.head()

#%%
df['duplicates'].value_counts()

#%%
# now we filter out images that don't have duplicates
df_undup = df[df['duplicates'] == 'unduplicated']
df_undup.shape

#%%
df['dx'].value_counts()

#%%
from sklearn.model_selection import train_test_split
# now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
y = df_undup['dx']
_, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
df_val.shape
# This set will be df_original excluding all rows that are in the val set
def get_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df['image_id'])
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'

# identify train and val rows
# create a new colum that is a copy of the image_id column
df['train_or_val'] = df['image_id']
# apply the function to this new column
df['train_or_val'] = df['train_or_val'].apply(get_val_rows)
# filter out train rows
df_train = df[df['train_or_val'] == 'train']
print(len(df_train))
print(len(df_val))

#%%
# Copy fewer class to balance the number of 7 classes
data_aug_rate = [15,10,5,50,0,40,5]
for i in range(7):
    if data_aug_rate[i]:
        df_train=df_train.append([df_train.loc[df_train['dx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
print(df_train['dx'].value_counts())

#%%
from glob import glob
base_skin_dir = os.path.join('..', 'input')

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

# %%
