from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from .grid import GRID


class Circles(object):
    def __init__(self):
        self.X, self.labels = make_circles(n_samples=400, noise=0.1, random_state=5622, factor=0.8)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class DataBlobs:
    def __init__(self, centers, std=1.75):
        self.X, self.labels = make_blobs(n_samples=400, n_features=2, cluster_std=std, centers=centers,
                                         shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class DataMoons(object):
    def __init__(self):
        self.X, self.labels = make_moons(n_samples=400, noise=0.05, shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


def get_inception_layer(f1, f2, f3):
    """
    Returns an "naive" inception block. Works better when we want to detect pixel level details
    :param f1: num filters for 1x1 convolution
    :param f2: num filters for 2x2 convolution
    :param f3: num filters for 4x4 convolution
    :return:
    """
    from keras import layers
    class InceptionLayer(layers.Layer):
        def __init__(self, f1, f2, f3):
            super(InceptionLayer, self).__init__()

            self.conv1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')
            self.conv2 = layers.Conv2D(f2, (2, 2), padding='same', activation='relu')
            self.conv4 = layers.Conv2D(f3, (4, 4), padding='same', activation='relu')
            self.pool = layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same')
            self.concat = layers.Concatenate(axis=-1)

        def call(self, inputs, *args, **kwargs):
            return self.concat([self.conv1(inputs),
                                self.conv2(inputs),
                                self.conv4(inputs),
                                self.pool(inputs)])

    return InceptionLayer(f1, f2, f3)
