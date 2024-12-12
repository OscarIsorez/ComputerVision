# example of loading the cifar10 dataset
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical


def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# plot first few images
def plot_images(images, labels):
    # define subplot
    pyplot.figure(figsize=(10, 10))
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        pyplot.imshow(images[i])
        # show the figure
        pyplot.show()


trainX, trainY, testX, testY = load_dataset()

plot_images(trainX, trainY)
