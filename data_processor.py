import tensorflow as tf
import numpy as np

from keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tdfs
from keras.preprocessing import sequence


def process(raw_text, vocabulary_size):
    # We might want to delete some words - maybe delete punctuation marks and words like "I, and". nltk library can help

    # Initialize the Tokenizer
    t = Tokenizer(num_words=vocabulary_size, oov_token=True)

    # Fit the tokenizer on text
    t.fit_on_texts(raw_text)

    # Turn the input into sequences of integers
    data = t.texts_to_sequences(raw_text)

    word_to_index = t.word_index
    index_to_word = {i: word for word, i in word_to_index.items()}

    return data, word_to_index, index_to_word, t


def load_data(train_or_test):
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


def load_all_data(vocabulary_size, max_words):  # This function is used by LSTM_with_tf_scan.py, which we aren't using
    # Loading the data in the given format, x -> list of strings, y -> list of integers
    x_train, y_train = load_data('train')
    x_test, y_test = load_data('test')

    # Transform the data integer format and get the necessary tools to deal with that later
    x_train, word_to_index, index_to_word, t = process(x_train, vocabulary_size)
    x_test = t.texts_to_sequences(x_test)

    # Finding out the longest and shortest word sequence in our reviews
    # print('Maximum review length: {}'.format(len(max((x_train + x_test), key=len))))  # 2493
    # print('Minimum review length: {}'.format(len(min((x_train + x_test), key=len))))  # 6

    # Limiting the maximum review length to max_words by truncating longer reviews
    # and padding shorter reviews with null value
    x_train = sequence.pad_sequences(x_train, maxlen=max_words)
    x_test = sequence.pad_sequences(x_test, maxlen=max_words)

    return x_train, y_train, x_test, y_test


def load_all_data_cell(vocabulary_size, max_words):  # Duplicated code, but I didn't want to change main.py
    x_train, y_train = load_data('train')
    x_test, y_test = load_data('test')

    x_train, word_to_index, index_to_word, t = process(x_train, vocabulary_size)
    x_test = t.texts_to_sequences(x_test)

    # print('Maximum review length: {}'.format(len(max((x_train + x_test), key=len))))  # 2493
    # print('Minimum review length: {}'.format(len(min((x_train + x_test), key=len))))  # 6

    x_train = sequence.pad_sequences(x_train, maxlen=max_words)
    x_test = sequence.pad_sequences(x_test, maxlen=max_words)

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    return x_train, y_train, x_test, y_test, word_to_index, index_to_word, t


# Define the encoder function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encoded = np.zeros((n_labels, n_unique_labels))  # now its [0 0] [0 0]
    one_hot_encoded[np.arange(n_labels), labels] = 1  # now its [1 0] [0 1]
    return one_hot_encoded


def load_imdb_data(num_words, maxlen):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words,
                                                                            maxlen=maxlen)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)  # padding='post'
    # print(len(x_train[101]) - np.count_nonzero(x_train[101] == 0))  # Count of nonpadded symbols
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    return x_train, y_train, x_test, y_test
