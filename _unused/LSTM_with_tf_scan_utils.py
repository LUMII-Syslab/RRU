from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tdfs
from tensorflow.keras.preprocessing import sequence


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


