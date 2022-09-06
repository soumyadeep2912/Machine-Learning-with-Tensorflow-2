import numpy as np
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def model():
    input = tf.keras.Input(shape=(28, 28, 1))
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5,
                                   activation='relu', padding='same')(input)
    maxpool2 = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=None, padding='valid')(conv1)
    droupout1 = tf.keras.layers.Dropout(0.25)(maxpool2)


    conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=5,
                                   activation='relu', padding='valid')(maxpool2)
    maxpool3 = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=None, padding='valid')(conv3)
    droupout2 = tf.keras.layers.Dropout(0.25)(maxpool3)

    flat = tf.keras.layers.Flatten()(droupout2)
    fc1 = tf.keras.layers.Dense(units=240, activation='relu')(flat)
    fc2 = tf.keras.layers.Dense(units=128, activation='relu')(fc1)
    final = tf.keras.layers.Dense(units=10, activation='softmax')(fc2)

    model = tf.keras.Model(inputs=input, outputs=final)

    loss = tf.keras.losses.CategoricalCrossentropy()

    model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = X_train.reshape((60000,28,28,1))
    X_test = X_test.reshape((10000,28,28,1))
    y_train = tf.squeeze(tf.one_hot(y_train, depth=10))
    y_test = tf.squeeze(tf.one_hot(y_test, depth=10))

    model = model()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
    lrr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

    model.fit(datagen.flow(X_train, y_train,
              batch_size=100), epochs=30, validation_data=(X_test, y_test))  # , callbacks=[early_stopping])
    model.save('mnist.h5')
