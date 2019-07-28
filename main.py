import os

import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import custom_model
import train
import util


def read_metadata():
    # Read data from csv
    df = pd.read_csv(util.METADATA_PATH)
    return df


def extract_labels(metadata):
    # Extract labels as categories from main dataframe.
    Y_orig = metadata['dx'].astype('category')
    print("Unique labels: ", Y_orig.unique())
    return Y_orig


def one_hot_encode_labels(Y_orig):
    # Convert catego1ries to integers
    Y_cat = Y_orig.cat.codes
    # Convert to one hot vector
    Y = to_categorical(Y_cat)
    return Y


def load_train_data(image_dir, target_size=(224, 224, 3)):
    print("Reading raw data...")
    images = []
    for image_path in os.listdir(image_dir):
        img = image.load_img(os.path.join(
            image_dir, image_path), target_size=target_size)
        images.append(image.img_to_array(img))
    X_orig = np.asarray(images)
    # Free up memory from images
    del images
    return X_orig


def split_train_test(X_orig, Y_orig):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_orig, Y_orig, test_size=0.05, random_state=50)
    return X_train, Y_train, X_test, Y_test


def normalize_train_test(X_train, X_test):
    # Divide by 255 to normalize pixel values
    X_train /= 255
    X_test /= 255

    # Subtract pixel mean
    mean = np.mean(X_train)
    X_train -= mean
    X_test -= mean

    return X_train, X_test


def main():
    df = read_metadata()
    print(df.head())

    Y_orig = extract_labels(df)
    X_orig = load_train_data(util.IMAGE_PATH, target_size=(112, 112, 3))

    X_train, Y_train, X_test, Y_test = split_train_test(X_orig, Y_orig)
    X_train, X_test = normalize_train_test(X_train, X_test)

    Y_train = one_hot_encode_labels(Y_train)
    Y_test = one_hot_encode_labels(Y_test)

    print('Shape of training data: ', X_train.shape)
    print('Shape of training labels: ', Y_train.shape)
    print('Shape of test data: ', X_test.shape)
    print('Shape of test labels: ', Y_test.shape)
    # Make user enter the epoch size
    num_epochs = 1
    while True:
        try:
            num_epochs = input("Please enter number of epochs: ")
            if int(num_epochs) <= 0:
                print("Number should be greater than 0.")
            else:
                break
        except ValueError:
            print("You must enter a valid integer.")

    num_epochs = int(num_epochs)

    # Train the model
    model = custom_model.load_pretrained_model(
        input_shape=X_train[0].shape, classes=Y_train.shape[1])
    train.train(model, X_train, Y_train, X_test, Y_test,
                num_epochs=num_epochs, batch_size=16, data_augmentation=True)


if __name__ == '__main__':
    main()
