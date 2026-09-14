import numpy as np


def load_mnist():
    # loads mnist dataset to pytorch dataloader
    f = np.load("definitely_mnist.npz")
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]
    f.close()

    return (x_train, y_train), (x_test, y_test)
