import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import sequence


def load_data(num_words, trim_length):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)

    # Get the longest sequence length
    maxlen = len(max((x_train + x_test), key=len))
    # Trim the sequences if such number was specified and it was less than maximum length
    if trim_length is not None:
        maxlen = min(maxlen, trim_length)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    return x_train, y_train, x_test, y_test, maxlen


# Define the encoder function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encoded = np.zeros((n_labels, n_unique_labels))  # Now it's [0 0] [0 0]
    one_hot_encoded[np.arange(n_labels), labels] = 1  # Now it's [1 0] [0 1]
    return one_hot_encoded


def get_sequence_lengths(sequences):  # Full sequence length minus the padded zeros
    sequence_lengths = []
    for sequence_ in sequences:
        sequence_ = np.array(sequence_)  # Without this line it returns full length for casual lists [1,1,1,0,0]->5
        sequence_length = len(sequence_) - np.count_nonzero(sequence_ == 0)  # Count of non-padded symbols
        sequence_lengths.append(sequence_length)
    return sequence_lengths
