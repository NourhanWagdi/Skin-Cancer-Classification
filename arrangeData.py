
#%%
import pandas as pd
import os
import util

# #%%
# df= pd.read_csv(util.METADATA_PATH)
# print(df.head())
# lesion_type_dict = {
#     'nv': 'Melanocytic nevi',
#     'mel': 'dermatofibroma',
#     'bkl': 'Benign keratosis-like lesions ',
#     'bcc': 'Basal cell carcinoma',
#     'akiec': 'Actinic keratoses',
#     'vasc': 'Vascular lesions',
#     'df': 'Dermatofibroma'
# }

# # %%
# df['cell_type'] = df['dx'].map(lesion_type_dict.get)
# df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
# df.head()



# # %%
# df_undup = df.groupby('lesion_id').count()
# df_undup = df_undup[df_undup['image_id'] == 1]
# df_undup.reset_index(inplace=True)
# df_undup.head()


# # %%
# def get_duplicates(x):
#     unique_list = list(df_undup['lesion_id'])
#     if x in unique_list:
#         return 'unduplicated'
#     else:
#         return 'duplicated'

# # create a new colum that is a copy of the lesion_id column
# df['duplicates'] = df['lesion_id']
# # apply the function to this new column
# df['duplicates'] = df['duplicates'].apply(get_duplicates)
# df.head()


# # %%
# df['duplicates'].value_counts()


# # %%
# df_undup = df[df['duplicates'] == 'unduplicated']
# df_undup.shape


# # %%
# # Train Test Split Data
# from sklearn.model_selection import train_test_split
# _, df_test = train_test_split(df_undup,test_size=0.2, random_state=42)
# df_test.shape


# # %%
# def get_val_rows(x):
#     test_list = list(df_test['image_id'])
#     if str(x) in test_list:
#         return 'test'
#     else:
#         return 'train'

# df['train_or_test'] = df['image_id']
# df['train_or_test'] = df['train_or_test'].apply(get_val_rows)
# df_train = df[df['train_or_test'] == 'train']
# df_test =df[df['train_or_test'] == 'test']


# # %%
# df_train.shape


# # %%
# df_test.shape


# # %%
# # Split test data into validation data
# df_test, df_val = train_test_split(df_test, test_size=0.3, random_state=101)
# df_val.shape

# #%%
# df_val['dx'].value_counts()

# #%%
# df_train['dx'].value_counts()

# #%%
# df_test['dx'].value_counts()

# #%%
# df_test['dx'].iloc[0]
# #%%
# # Transfer images from source to destination
# df_train.head()

# from shutil import copy
# def copy_data(df, src, dest):
#     images = list(df['image_id'])
#     i = 0
#     for image in images:
#         fname = image + ".jpg" 
#         tempsrc = os.path.join(src, fname)
#         temp = os.path.join(df['dx'].iloc[i],fname)
#         tempdest = os.path.join(dest, temp)
#         copy(tempsrc, tempdest)
#         i += 1
# #%%
src = util.IMAGE_PATH
dest = util.SKIN_CANCER_HAMNIST_HAM_1000_PATH
dest = os.path.join(dest, 'arrangedData')
traindest = os.path.join(dest, 'training')
testdest = os.path.join(dest, 'testing')
valdest = os.path.join(dest, 'validation')

print(traindest)
print(testdest)
print(valdest)

# #%%
# copy_data(df_val, src, valdest)

# #%%
# copy_data(df_test, src, testdest)

# #%%
# copy_data(df_train, src, traindest)

# #%%
aug_dir = 'aug_dir'
# os.mkdir(os.path.join(dest, aug_dir))
#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from shutil import rmtree, copy
aug_dir_path = os.path.join(dest, aug_dir)

class_list = ['mel','bkl','bcc','akiec','vasc','df']

for item in class_list:
    #os.mkdir(aug_dir_path)
    temp_dir_path = os.path.join(aug_dir_path,'temp')
    os.mkdir(temp_dir_path)
    
    itempath = os.path.join(traindest, item)
    images = os.listdir(itempath)

    for fname in images:
        src = os.path.join(itempath, fname)
        dst = os.path.join(temp_dir_path, fname)
        copy(src,dst)

    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(
        aug_dir_path,
        save_to_dir=itempath,
        save_format='jpg',
        target_size=(224,224),
        batch_size=batch_size
    )

    num_aug_images = 6000
    num_files = len(os.listdir(temp_dir_path))
    num_batches = int(np.ceil((num_aug_images - num_files) / batch_size))

    for i in range(0, num_batches):
        imgs, lbls = next(aug_datagen)
    

    rmtree(aug_dir_path)