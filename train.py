import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tempfile
from network import *
from preprocess import datas
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from threading import Thread, Lock, currentThread


def to_image(path):
    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [WIDTH, HEIGHT])

    return image


lock = Lock()

total = 0
paths = []
labels = []


def gen_ds(proj):
    global paths, labels, total

    print(proj)

    proj.calc()

    loc_paths = sorted([str(p) for p in proj.frames.glob("**/*.jpg")])

    cut = np.loadtxt(proj.txt).astype(np.int16)

    loc_images = []
    loc_labels = []
    sequences = np.split(loc_paths, cut[1:])
    for sequence in sequences:
        for i in range(0, len(sequence), 2):
            j = i + 1
            if j == len(sequence):
                break
            loc_images.append([loc_paths[i], loc_paths[j]])
            loc_labels.append(0)

    for i in range(0, len(sequences)):
        j = i + 1
        if j == len(sequences):
            break
        loc_images.append([sequences[i][-1], sequences[j][0]])
        loc_labels.append(1)

    for i in range(loc_labels.count(0) - loc_labels.count(1)):
        j = randint(len(sequences))
        k = randint(len(sequences))
        while k == j:
            k = randint(len(sequences))
        m = randint(len(sequences[j]))
        n = randint(len(sequences[k]))
        path1 = sequences[j][m]
        path2 = sequences[k][n]
        loc_images.append([path1, path2])
        loc_labels.append(1)

    loc_total = len(loc_labels)
    print(f"name: {currentThread().getName()}, local_total: {loc_total}")

    with lock:
        paths += loc_images
        labels += loc_labels

        total += loc_total


threads = []
for proj in datas[3:12]:
    print("path", proj.path)
    t = Thread(target=gen_ds, args=(proj,), name=proj.path.name)
    threads.append(t)
    t.start()
for t in threads:
    t.join()

print("constructing the dataset...")

path_ds = tf.data.Dataset.from_tensor_slices(paths)
label_ds = tf.data.Dataset.from_tensor_slices(labels)

print("standlizing the dataset...")

dataset = (
    tf.data.Dataset.zip((path_ds, label_ds))
    .shuffle(buffer_size=total, reshuffle_each_iteration=True)
    .repeat()
    .map(
        lambda paths, label: (
            tf.convert_to_tensor((to_image(paths[0]), to_image(paths[1]))),
            label,
        )
    )
)


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

_BATCH_SIZE = 2

dataset = (
    dataset.map(
        lambda images, label: (
            tf.convert_to_tensor((images[0] / 255.0, images[1] / 255.0)),
            label,
        )
    )
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    .batch(_BATCH_SIZE, drop_remainder=True)
)

print(dataset)

_EPOCHS = 5

print("batching...")

model = make_model()

initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")

# model.load_weights(initial_weights)

steps = int(total / _BATCH_SIZE)

train_steps = int(steps * 0.7)
val_steps = int(steps * 0.3)

train_ds = dataset.take(train_steps * _EPOCHS)
val_ds = dataset.take(train_steps * _EPOCHS)

print(train_ds)
print(val_ds)

print("fitting")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=_EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)
model.save_weights(initial_weights)

# results
acc = history.history["acc"]
val_acc = history.history["val_acc"]

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
