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
df = pd.read_csv(util.METADATA_PATH)
print(df.head())


# %%
df.isnull().sum()


# %%
df['age'].fillna(df['age'].mean(), inplace=True)


# %%
stats = df['dx'].value_counts()


# %%
# import seaborn as sns


# %%
plt.figure(figsize=(10,6))
plt.title('Count of data based on catoegory.')
# sns.barplot(x=stats.index, y=stats)
plt.ylabel("Number of samples.")


# %%
dx_type_stats = df['dx_type'].value_counts()


# %%
sns.barplot(x=dx_type_stats.index, y=dx_type_stats)


# %%
localization_stats = df['localization'].value_counts()


# %%
# sns.barplot(x=localization_stats.index, y=localization_stats)


# %%
sex_stats = df['sex'].value_counts()
# sns.barplot(x=sex_stats.index, y=sex_stats)


# %%
df.head()


# %%
df['dx_type'].unique()


# %%
df['localization'].unique()


# %%
dx_type_dict = {
    'histo': 0,
    'consensus': 1,
    'confocal': 2,
    'follow_up': 3
}

localization_dict = {
    'scalp': 0,
    'ear': 1,
    'face': 1,
    'back': 3,
    'trunk': 4,
    'chest': 5,
    'upper extremity': 6,
    'abdomen': 7,
    'lower extremity': 6,
    'genital': 2,
    'neck': 8,
    'hand': 9,
    'foot': 10,
    'acral': 11,
    'unknown': 12
}

df['localization'] = df['localization'].map(localization_dict)
df['dx_type'] = df['dx_type'].map(dx_type_dict)
df.head()


# %%
sex_map = {
    'male': 0,
    'female': 1
}
df['sex'] = df['sex'].map(sex_map)
df.head()


# %%
samimg = image.load_img(os.path.join(util.IMAGE_PATH, df['image_id'][0] + '.jpg'))


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



# %%
samimg.resize((100,75))


# %%
df_undup['image'] = df_undup['image_id'].map(
    lambda x: np.asarray(image.load_img(
        os.path.join(util.IMAGE_PATH, x + ".jpg")
    ).resize((224,224)))
)


# %%
df = df_undup
df_undup.shape


# %%
df['image'] = df['image']/255
df.head()


# %%
df['dx'].value_counts()


# %%
X = df[['dx_type', 'age', 'sex', 'localization','image']]


# %%
X.head()


# %%
Y = df['dx']


# %%
Y.head()


# %%
Y.unique()


# %%
ymap = {
    'bkl':0, 
    'nv':1, 
    'df':2, 
    'mel':3, 
    'vasc':4, 
    'bcc':5, 
    'akiec':6
}

Y = Y.map(ymap)
Y.head()


# %%
# Train Test Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=42)


# %%
X_train.count()


# %%
# Normalize data
X_train = np.asarray(X_train['image'].to_list())
X_test = np.asarray(X_test['image'].to_list())


# %%
X_train.shape


# %%
X_test.shape


# %%
train_mean = np.mean(X_train)
train_std = np.std(X_train)

X_train = (X_train - train_mean) / train_std


# %%
from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(Y_train, num_classes=7)


# %%
Y_train.shape


# %%
# Separate validation test
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)


# %%
from custom_model import naiveModel, get_inception_model
model = get_inception_model(input_shape=(224,224,3))


# %%
from train import train
history = train(model, X_train, Y_train, X_val, Y_val, 50, 8, False)


# %%
type(history.history)


# %%
Y_test = to_categorical(Y_test, num_classes=7)


# %%
_, test_acc = model.evaluate(X_test, Y_test)
print("Accuracy on the test set = ", str(test_acc * 100))


# %%
history_df = pd.DataFrame([history.history['acc'], history.history['val_acc']])


# %%
# sns.lineplot(data=history_df)


# %%



