import numpy as np


def load_mnist(test_size: float = 0.2):
    with np.load("definitely_mnist.npz") as f:
        x, y = f["x_train"], f["y_train"]

    rows = np.arange(x.shape[0])
    np.random.shuffle(rows)
    train_size = int(x.shape[0] * (1 - test_size))
    x_train, y_train = x[rows[:train_size]], y[rows[:train_size]]
    x_test, y_test = x[rows[train_size:]], y[rows[train_size:]]
    return (x_train, y_train), (x_test, y_test)
