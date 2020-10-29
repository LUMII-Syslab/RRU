import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence


def process(raw_text, vocabulary_size):
    # We might want to delete some words - maybe delete punctuation marks and words like "I", "and", etc.
    # "nltk" library can help with that, if you wish

    # Initialize the Tokenizer
    t = Tokenizer(num_words=vocabulary_size, oov_token=True)

    # Fit the tokenizer on text
    t.fit_on_texts(raw_text)

    # Turn the input into sequences of integers
    data = t.texts_to_sequences(raw_text)

    word_to_index = t.word_index
    index_to_word = {i: word for word, i in word_to_index.items()}

    return data, word_to_index, index_to_word, t


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


def load_data_tdfs(vocabulary_size):
    import tensorflow_datasets as tdfs

    # If you want to use it in your program, you need to use:
    # from imdb_utils import load_data_tdfs
    # X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, WORD_TO_ID, ID_TO_WORD, T, max_sequence_length = load_data_tdfs(vocabulary_size)

    def load_data_slice(train_or_test):
        # We don't use the dataset for epochs, so this will just give us the imdb_reviews in some random shuffled order
        ds = tdfs.load('imdb_reviews', split=train_or_test, shuffle_files=True)
        # ds = ds.take(1000)  # To take only a few examples
        x = []
        y = []
        for review in tdfs.as_numpy(ds):
            text = review['text']
            text = text.decode()  # bytes -> string
            label = review['label']  # 0 for negative, 1 for positive
            x.append(text)
            y.append(label)
        return x, y

    x_train, y_train = load_data_slice('train')
    x_test, y_test = load_data_slice('test')

    x_train, word_to_index, index_to_word, t = process(x_train, vocabulary_size)
    x_test = t.texts_to_sequences(x_test)

    # Get the longest sequence length
    max_words = len(max((x_train + x_test), key=len))

    x_train = sequence.pad_sequences(x_train, maxlen=max_words, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_words, padding='post')

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    return x_train, y_train, x_test, y_test, word_to_index, index_to_word, t, max_words
