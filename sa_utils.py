# This file contains helper functions that are related to sentiment analysis, such as loading data

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence


def load_data(num_words, trim_length):
    """
        This function loads the IMDB data set.

        Input:
            num_words: int, number of words allowed in the "IMDB" data set vocabulary;
            trim_length: int, the maximum length of sequences.

        Output:
            x_train: list, input sequences of words (represented by integers) for training;
            y_train: list, output sequences of one-hot encoded labels for training;
            x_test: list, input sequences of words (represented by integers) for testing;
            y_test: list, output sequences of one-hot encoded labels for testing;
            maxlen: int, the maximum length of sequences.
    """

    # Load the IMDB data set from TensorFlow
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)

    # Get the longest sequence in the input data
    maxlen = len(max((x_train + x_test), key=len))

    # Trim the sequences if such number was specified and it was less than the maximum length
    if trim_length is not None:
        maxlen = min(maxlen, trim_length)

    # Pad the sequences if shorter, or trim the sequences if longer
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')

    # One-hot encode the labels, that is [0] -> [1, 0] and [1] -> [0, 1]
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return x_train, y_train, x_test, y_test, maxlen


def one_hot_encode(labels):
    """
        This function one-hot encodes the labels.

        Input:
            labels: list, a list of integers representing some answers.

        Output:
            one_hot_encoded: list, one-hot encoded labels.
    """

    # How many labels are there?
    n_labels = len(labels)

    # How many different labels are there?
    n_unique_labels = len(np.unique(labels))

    # Create a NumPy array in shape [number of labels, number of unique labels]
    one_hot_encoded = np.zeros((n_labels, n_unique_labels))  # Now each data sample is [0, 0] (in case of IMDB data set)

    # Put ones where necessary (one 1 for each data sample)
    one_hot_encoded[np.arange(n_labels), labels] = 1  # Now data samples are [1, 0] or [0, 1] (in case of IMDB data set)

    return one_hot_encoded


# Function that returns sequence lengths minus the padded zeros
def get_sequence_lengths(sequences):
    """
        This function returns the lengths of the sequences (the length without the padded zeros).

        Input:
            sequences: list, a list of sequences.

        Output:
            sequence_lengths: list, a list of sequence lengthsone-hot encoded labels.
    """

    sequence_lengths = []

    # Go through each of the sequences
    for sequence_ in sequences:
        # Without this line it returns full length for lists (if we don't use NumPy arrays), so we make the type as a
        # numpy array. Without this line if [1, 1, 1, 0, 0] is passed, it returns 5, but we want it to return 3.
        sequence_ = np.array(sequence_)

        # Count of non-padded symbols
        sequence_length = len(sequence_) - np.count_nonzero(sequence_ == 0)

        # Add the sequence length to the corresponding list
        sequence_lengths.append(sequence_length)
    return sequence_lengths
