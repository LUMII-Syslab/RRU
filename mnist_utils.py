# This file contains helper functions that are related to mnist sequence classification, such as loading data

import numpy as np
import tensorflow as tf

supported_data_sets = ["SequentialMNIST", "P-MNIST"]


def binary_mask(numbers, mask_size=10):
    """
        This function binary masks the passed list of numbers.

        Input:
            numbers: list, a list of numbers, for example, [1, 2, 3];
            mask_size: int, total numbers possible in this mask, for example, for MIDI numbers – 10.

        Output:
            masks: list, a list of masks (numbers masked in a binary mask).

        Example:
            Input:
                numbers = [1, 2, 3]
                mask_size = 4
            Output:
                masks = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    """

    masks = []

    # Go through each of the numbers and put a one in number'th place in the mask array
    for number in numbers:
        # Create an array of length [mask_size], filled with zeros
        mask = np.zeros(mask_size)

        mask[number] = 1

        masks.append(mask)

    return masks


# Loads the data set with the passed name
def load_data(name):
    """
        This function loads and returns the requested data set.

        Input:
            name: string, the name of the data set.

        Output:
            x_train: list, input sequences of words (represented by integers) for training;
            y_train: list, output sequences of one-hot encoded labels for training;
            x_test: list, input sequences of words (represented by integers) for testing;
            y_test: list, output sequences of one-hot encoded labels for testing;
            maxlen: int, the maximum length of sequences.
    """

    print("Started loading data...")

    # Check if data set is supported
    if name not in supported_data_sets:
        raise Exception("This code doesn't support the following data set!")

    seed = 0
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = np.reshape(x_train, (-1, 784, 1))
    x_test = np.reshape(x_test, (-1, 784, 1))

    y_train = binary_mask(y_train)
    y_test = binary_mask(y_test)

    x_train = x_train / 255
    x_test = x_test / 255

    if name == "P-MNIST":
        perm = rng.permutation(x_train.shape[1])
        x_train = x_train[:, perm]
        x_test = x_test[:, perm]

    # Reserve 10,000 samples for validation.
    x_valid = x_train[-10000:]
    y_valid = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


if __name__ == '__main__':  # Main function
    # To load a specific data set use:
    X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, X_TEST, Y_TEST = load_data("SequentialMNIST")
