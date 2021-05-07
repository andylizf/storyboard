import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
import tensorflow_addons as tfa

METRICS = [
    tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1, name="mcc"),
    TruePositives(name="tp"),
    FalsePositives(name="fp"),
    TrueNegatives(name="tn"),
    FalseNegatives(name="fn"),
    "acc",
    Precision(name="precision"),
    Recall(name="recall"),
    AUC(name="auc"),
    AUC(name="prc", curve="PR"),  # precision-recall curve
]


WIDTH = 224
HEIGHT = 224


def make_model(output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    model = keras.Sequential(
        [
            keras.applications.MobileNet(include_top=False, input_shape=(224, 224, 3)),
            GlobalAveragePooling2D(),
            MaxPooling2D((2, 2)),
            Dense(64, activation=LeakyReLU()),
            Dropout(0.5),
            Dense(32, activation=LeakyReLU()),
            Dropout(0.5),
            Dense(1, activation="sigmoid", bias_initializer=output_bias),
        ]
    )
    model.summary()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=METRICS,
    )

    return model
