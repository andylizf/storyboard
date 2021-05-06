import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import tempfile
from network import *
from preprocess import DataProject, datas
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread, Lock, currentThread
from tensorflow.math import argmax


def to_image(path):
    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [WIDTH, HEIGHT])
    image /= 255.0

    return image


label0 = 0
label1 = 0
total = 0

lock = Lock()
COUNT = 100000

paths = []
labels = []


def gen_ds(proj):
    global paths, labels, label0, label1, total

    print(proj)

    proj.calc()

    loc_paths = [str(p) for p in proj.flows.glob("**/*.jpg")]

    loc_labels = np.zeros(len(loc_paths)).astype(np.bool)
    cut = np.loadtxt(proj.txt).astype(np.int16)
    loc_labels[cut] = True

    loc_total = len(loc_paths) / LENGTH
    loc_label1 = len(cut)
    loc_label0 = loc_total - loc_label1

    print(
        f"name: {currentThread().getName()}, local_total: {loc_total}, local_label1: {loc_label1}, local_label0: {loc_label0}"
    )

    with lock:
        paths += loc_paths
        labels = np.concatenate((labels, loc_labels))

        label0 += loc_label0
        label1 += loc_label1
        total += loc_total


threads = []
for proj in datas[1:2]:
    t = Thread(target=gen_ds, args=(proj,), name=proj.path.name)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("constructing the dataset...")

path_ds = tf.data.Dataset.from_tensor_slices(paths)
image_ds = path_ds.map(to_image)

label_ds = tf.data.Dataset.from_tensor_slices(labels)

print("standlizing the dataset...")

_BATCH_SIZE = 2

dataset = (
    tf.data.Dataset.zip((image_ds, label_ds))
    .batch(LENGTH, drop_remainder=True)
    .map(lambda x, y: (x, argmax(y) != 0 and argmax(y) != LENGTH - 1))
    .batch(_BATCH_SIZE, drop_remainder=True)
)

print(f"total: {total}, label1: {label1}, label0: {label0}")

val_ds = dataset.take(int(total / _BATCH_SIZE * 0.3))
train_ds = dataset.skip(int(total / _BATCH_SIZE * 0.3))

_EPOCHS = 5

print("batching...")

print(train_ds)
print(val_ds)

initial_bias = np.log([label1 / label0])
print(initial_bias)

model = make_model(initial_bias)

initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")

# model.load_weights(initial_weights)

class_weight = {0: (1 / label0) * (total) / 2, 1: (1 / label1) * (total) / 2}
print(class_weight)

print("fitting")
history = model.fit(
    train_ds, validation_data=val_ds, epochs=_EPOCHS, class_weight=class_weight
)

model.save_weights(initial_weights)

# results
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(_EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()
