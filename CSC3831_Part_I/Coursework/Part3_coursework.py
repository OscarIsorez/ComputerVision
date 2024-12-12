# example of loading the cifar10 dataset
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)


def load_dataset():
    # load dataset
    train, test = cifar10.load_data()
    trainX, trainY = train
    testX, testY = test

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# plot first few images
def plot_images(images, labels):
    f, ax = pyplot.subplots(4, 4)
    for i in range(16):
        ax[i // 4, i % 4].imshow(images[i])
        ax[i // 4, i % 4].set_title(labels[i])
        ax[i // 4, i % 4].axis("off")
    pyplot.show()


def prep_pixels(train, test):
    train_norm = train.astype("float32")
    test_norm = test.astype("float32")
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def build_model():

    size = 32
    depth = 3

    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            input_shape=(size, size, depth),
        )
    )
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title("Cross Entropy Loss")
    pyplot.plot(history.history["loss"], color="blue", label="train")
    pyplot.plot(history.history["val_loss"], color="orange", label="test")
    pyplot.subplot(212)
    pyplot.title("Classification Accuracy")
    pyplot.plot(history.history["accuracy"], color="blue", label="train")
    pyplot.plot(history.history["val_accuracy"], color="orange", label="test")
    filename = sys.argv[0].split("/")[-1]
    pyplot.savefig(filename + "_plot.png")
    pyplot.close()


def run_test_harness():
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = build_model()

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        trainX,
        trainY,
        epochs=100,
        batch_size=64,
        validation_data=(testX, testY),
        verbose=0,
    )
    _, acc = model.evaluate(testX, testY, verbose=0)

    print("> %.3f" % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)


run_test_harness()
