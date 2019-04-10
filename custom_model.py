from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, Input, MaxPooling2D)
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model


def apply_maxpool(X):
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    return X


def construct_VGG19(input_size=(224, 224, 3), classes=7):
    # Stack up the layers
    X_Input = Input(input_size)
    # Stage 1
    X = Conv2D(filters=64, kernel_size=(3, 3),
               strides=(1, 1), padding='same')(X_Input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=64, kernel_size=(3, 3),
               strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Apply Max Pool
    X = apply_maxpool(X)
    print(X.shape)

    # Stage 2
    for i in range(2):
        X = Conv2D(filters=128, kernel_size=(3, 3),
                   strides=(1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

    # Apply Max Pool
    X = apply_maxpool(X)
    print(X.shape)

    # Stage 3
    for i in range(4):
        X = Conv2D(filters=256, kernel_size=(3, 3),
                   strides=(1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
    # Apply Max Pool
    X = apply_maxpool(X)
    print(X.shape)

    # Stage 4
    for i in range(4):
        X = Conv2D(filters=512, kernel_size=(3, 3),
                   strides=(1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
    # Apply Max Pool
    X = apply_maxpool(X)
    print(X.shape)

    # Stage 5
    for i in range(4):
        X = Conv2D(filters=512, kernel_size=(3, 3),
                   strides=(1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
    # Apply Max Pool
    X = apply_maxpool(X)
    print(X.shape)

    # Flatten this layer
    X = Flatten()(X)

    # Dense Layers
    for i in range(2):
        X = Dense(4096, activation='relu')(X)

    # Last layers
    X = Dense(classes, activation='softmax')(X)

    # Create Model
    model = Model(inputs=X_Input, outputs=X)
    return model

# Unit Test


def main():
    model = construct_VGG19()
    #model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='vgg19.png')

# main()
