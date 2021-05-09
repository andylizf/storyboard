import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
import tensorflow_addons as tfa

WIDTH = 224
HEIGHT = 224
LENGTH = 2
SHAPE = [LENGTH, WIDTH, HEIGHT, 3]


def build_convnet(shape):
    momentum = 0.9
    model = keras.Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=shape, padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=momentum))

    # flatten...
    model.add(GlobalMaxPool2D())
    return model


def make_model(output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    convnet = build_convnet(SHAPE[1:])
    model = keras.Sequential(
        [
            TimeDistributed(convnet, input_shape=(LENGTH, 224, 224, 3)),
            GRU(64),
            Dense(1024, activation="relu"),
            Dropout(0.5),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid", bias_initializer=output_bias),
        ]
    )
    model.summary()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics="acc",
    )

    return model
