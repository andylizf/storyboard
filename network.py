import tensorflow as tf
from tensorflow.keras import *
import tensorflow_addons as tfa

METRICS = [
    tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1, name="mcc"),
    # metrics.TruePositives(name="tp"),
    # metrics.FalsePositives(name="fp"),
    # metrics.TrueNegatives(name="tn"),
    # metrics.FalseNegatives(name="fn"),
    "acc",
    # metrics.Precision(name="precision"),
    # metrics.Recall(name="recall"),
    # metrics.AUC(name="auc"),
    # metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
]


WIDTH = 224
HEIGHT = 224
LENGTH = 7


def make_model(output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    convnet = applications.MobileNet(include_top=False, input_shape=(224, 224, 3))
    model = Sequential(
        [
            layers.TimeDistributed(convnet, input_shape=(LENGTH, 224, 224, 3)),
            layers.TimeDistributed(layers.Flatten()),
            layers.GRU(128, activation=layers.LeakyReLU(), return_sequences=False),
            layers.Dense(64, activation=layers.LeakyReLU()),
            layers.Dropout(0.5),
            layers.Dense(32, activation=layers.LeakyReLU()),
            layers.Dropout(0.5),
            layers.Dense(1, bias_initializer=output_bias),
        ]
    )
    model.summary()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=METRICS,
    )

    return model
