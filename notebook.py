# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import util

from tensorflow.keras.preprocessing import image


# %%
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# %%
df = pd.read_csv(util.METADATA_PATH)
print(df.head())


# %%
df['cell_type'] = df['dx'].map(lesion_type_dict.get)
df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
df.head()


# %%
df_undup = df.groupby('lesion_id').count()
df_undup = df_undup[df_undup['image_id'] == 1]
df_undup.reset_index(inplace=True)
df_undup.head()


# %%
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


# %%
df['duplicates'].value_counts()


# %%
df_undup = df[df['duplicates'] == 'unduplicated']
df_undup.shape


# %%
# Train Test Split Data
from sklearn.model_selection import train_test_split
_, df_test = train_test_split(df_undup,test_size=0.2, random_state=42)
df_test.shape


# %%
def get_val_rows(x):
    test_list = list(df_test['image_id'])
    if str(x) in test_list:
        return 'test'
    else:
        return 'train'

df['train_or_test'] = df['image_id']
df['train_or_test'] = df['train_or_test'].apply(get_val_rows)
df_train = df[df['train_or_test'] == 'train']
df_test =df[df['train_or_test'] == 'test']


# %%
df_train.shape


# %%
df_test.shape


# %%
# Split test data into validation data
df_test, df_val = train_test_split(df_test, test_size=0.3, random_state=101)
df_val.shape


# %%
df_test.shape


# %%
df_train['dx'].value_counts()

# %%
# Function to load images.
def load_img(df):
    df['image'] = df['image_id'].map(
        lambda x: np.asarray(image.load_img(
            os.path.join(util.IMAGE_PATH, x + ".jpg")
        ).resize((224,224)))
    )
    return df


# %%
df_train = load_img(df_train)
df_test = load_img(df_test)
df_val = load_img(df_val)


# %%
# Copy fewer class to balance the number of 7 classes
data_aug_rate = [15,10,5,50,0,40,5]
for i in range(7):
    if data_aug_rate[i]:
        df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1),                                                 ignore_index=True)


# %%
df_train['dx'].value_counts()


# %%
Y_train = df_train['dx']


# %%
Y_test = df_test['dx']
Y_val = df_val['dx']


# %%
Y_val.head()


# %%
# Normalize data
X_train = np.asarray(df_train['image'].to_list())
X_test = np.asarray(df_test['image'].to_list())
X_val = np.asarray(df_val['image'].to_list())


# %%
X_train.shape


# %%
X_test.shape


# %%
train_mean = np.mean(X_train)
train_std = np.std(X_train)

X_train = (X_train - train_mean) / train_std

ymap = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}

Y_train = Y_train.map(ymap.get)
Y_test = Y_test.map(ymap.get)
Y_val = Y_val.map(ymap.get)

# %%
from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(Y_train, num_classes=7)
Y_test = to_categorical(Y_test, num_classes=7)
Y_val = to_categorical(Y_val, num_classes=7)


# %%
Y_train.shape


# %%
from custom_model import naiveModel, get_inception_model
model = get_inception_model(input_shape=(224,224,3))


# %%
from train import train
history = train(model, X_train, Y_train, X_val, Y_val, 30, 32, True)

# %%
_, test_acc = model.evaluate(X_test, Y_test)
print("Accuracy on the test set = ", str(test_acc * 100))



