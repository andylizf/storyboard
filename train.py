import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

    loc_paths = sorted([str(p) for p in proj.frames.glob("**/*.jpg")])

    cut = np.loadtxt(proj.txt).astype(np.int16)

    loc_total = int(len(loc_paths) / 2)

    loc_images = []
    loc_labels = []
    for i in range(0, loc_total * 2, 2):
        j, k = i, i + 1
        if k == len(loc_paths):
            break
        if j != 0 and j in cut:
            j, k = j - 1, k - 1
        loc_images.append([loc_paths[j], loc_paths[k]])
        loc_labels.append(k in cut)

    loc_label1 = len(cut)
    loc_label0 = loc_total - loc_label1

    print(
        f"name: {currentThread().getName()}, local_total: {loc_total}, local_label1: {loc_label1}, local_label0: {loc_label0}"
    )

    with lock:
        paths += loc_images
        labels += loc_labels

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
image_ds = path_ds.map(
    lambda paths: tf.convert_to_tensor((to_image(paths[0]), to_image(paths[1])))
)

label_ds = tf.data.Dataset.from_tensor_slices(labels)

print("standlizing the dataset...")

dataset = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(buffer_size=1500)


def test(cnt):
    plt.figure(figsize=(10, 10))
    i = 0
    for ((image1, image2), label) in dataset.take(cnt):
        ax = plt.subplot(1, 2, 1)
        plt.imshow(image1.numpy().astype("uint8"))
        plt.axis("off")
        ax = plt.subplot(1, 2, 2)
        plt.imshow(image2.numpy().astype("uint8"))
        plt.axis("off")
        plt.suptitle(f"label{int(label)}")
        plt.savefig(f"test{i}.jpg")
        i += 1


# test(40)


def scaling(image):
    return 2 * image / 255.0 - 1


_BATCH_SIZE = 2

dataset = (
    dataset.map(
        lambda images, label: (
            tf.convert_to_tensor((scaling(images[0]), scaling(images[1]))),
            label,
        )
    )
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    .batch(_BATCH_SIZE, drop_remainder=True)
)

print(dataset)
print(f"total: {total}, label1: {label1}, label0: {label0}")

_EPOCHS = 20

print("batching...")


initial_bias = np.log([label1 / label0])
print(initial_bias)

model = make_model(initial_bias)

initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")

# model.load_weights(initial_weights)

class_weight = {0: (1 / label0) * (total) / 2, 1: (1 / label1) * (total) / 3}
print(class_weight)

steps = int(total / _BATCH_SIZE)

train_steps = int(steps * 0.7)
val_steps = int(steps * 0.3)

train_ds = dataset.repeat().take(train_steps * _EPOCHS)
val_ds = dataset.repeat().take(train_steps * _EPOCHS)

print(train_ds)
print(val_ds)

print("fitting")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=_EPOCHS,
    class_weight=class_weight,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)
model.save_weights(initial_weights)
