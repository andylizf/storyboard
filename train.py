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
        for i in range(0, len(sequence), LENGTH):
            if i + LENGTH > len(sequence):
                break
            assert len(loc_paths[i : i + LENGTH]) == LENGTH
            loc_images.append(loc_paths[i : i + LENGTH])
            loc_labels.append(0)

    for i in range(0, len(sequences)):
        j = i + 1
        if j == len(sequences):
            break
        if len(sequences[i]) < DELTA or len(sequences[j]) < DELTA:
            continue
        assert len([*sequences[i][-DELTA:], *sequences[j][:DELTA]]) == LENGTH
        loc_images.append([*sequences[i][-DELTA:], *sequences[j][:DELTA]])
        loc_labels.append(1)

    for i in range(loc_labels.count(0) - loc_labels.count(1)):
        j = randint(len(sequences))
        k = randint(len(sequences))
        while j == k or len(sequences[j]) < DELTA or len(sequences[k]) < DELTA:
            j = randint(len(sequences))
            k = randint(len(sequences))
        m = randint(len(sequences[j]) - DELTA)
        n = randint(len(sequences[k]) - DELTA)
        assert (
            len([*sequences[j][m : m + DELTA], *sequences[k][n : n + DELTA]]) == LENGTH
        )
        loc_images.append([*sequences[j][m : m + DELTA], *sequences[k][n : n + DELTA]])
        loc_labels.append(1)

    loc_total = len(loc_labels)
    print(f"name: {currentThread().getName()}, local_total: {loc_total}")

    with lock:
        paths += loc_images
        labels += loc_labels

        total += loc_total


threads = []
for proj in datas[5:6]:
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
)

dataset = dataset.map(
    lambda path, label: (
        tf.convert_to_tensor(tf.map_fn(to_image, path, fn_output_signature=tf.float32)),
        label,
    )
)


def test(cnt):
    plt.figure(figsize=(10, 10))
    i = 0
    for (images, label) in dataset.take(cnt):
        for j in range(LENGTH):
            ax = plt.subplot(1, LENGTH, j)
            plt.imshow(images[j].numpy().astype("uint8"))
            plt.axis("off")
        plt.suptitle(f"label{int(label)}")
        plt.savefig(f"test{i}.jpg")
        i += 1


test(40)

_BATCH_SIZE = 2

dataset = (
    dataset.map(
        lambda images, label: (
            tf.convert_to_tensor(tf.map_fn(lambda image: image / 255.0, images)),
            label,
        )
    )
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    .batch(_BATCH_SIZE, drop_remainder=True)
)

_EPOCHS = 5

print("batching...")

model = make_model()

initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")

# model.load_weights(initial_weights)

steps = int(total / _BATCH_SIZE)

train_steps = int(steps * 0.7)
val_steps = int(steps * 0.3)

train_ds = dataset.take(train_steps * _EPOCHS)
val_ds = dataset.take(val_steps * _EPOCHS)

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
