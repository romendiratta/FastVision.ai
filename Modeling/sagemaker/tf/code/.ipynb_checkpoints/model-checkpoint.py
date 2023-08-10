import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


def get_model(learning_rate, weight_decay, optimizer, momentum, size):

    inputs = keras.Input((128, 128, 128, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", data_format='channels_last')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", data_format='channels_last')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", data_format='channels_last')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", data_format='channels_last')(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    if optimizer.lower() == 'sgd':
        opt = SGD(learning_rate=learning_rate * size, momentum=momentum)
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate * size)
    else:
        opt = Adam(learning_rate=learning_rate * size)

    model = keras.Model(inputs, outputs, name="3dcnn")

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=[
                      tf.keras.metrics.Accuracy(),
                      tf.keras.metrics.AUC(),
                      tf.keras.metrics.Precision(),
                      tf.keras.metrics.Recall()])

    return model