import numpy as np
import pickle
import zipfile

from sklearn.utils import shuffle

from tensorflow.keras.preprocessing.text import Tokenizer

PADDING = "C_PAD"
UNKNOWN = "C_UNK"


def prepare_enwik8(vocabulary_size, window_size, step_size):
    print("  Preparing enwik8...")
    prepare_enwik8_or_text8(name="enwik8", vocabulary_size=vocabulary_size, window_size=window_size, step_size=step_size)


def prepare_text8(vocabulary_size, window_size, step_size):
    print("  Preparing text8...")
    prepare_enwik8_or_text8(name="text8", vocabulary_size=vocabulary_size, window_size=window_size, step_size=step_size)


def prepare_enwik8_or_text8(name, vocabulary_size, window_size, step_size):
    # Read data set to a variable
    if name == "enwik8":
        data = zipfile.ZipFile("data/enwik8.zip").read(name)
    elif name == "text8":
        data = zipfile.ZipFile("data/text8.zip").read(name)
    else:
        raise Exception("    This function doesn't allow data sets, that aren't enwik8 or text8")

    print("    Length of {} data set is {}".format(name, len(data)))

    # Standard data set split is 90%/5%/5%, and 5% are 5 000 000
    num_test_chars = 5000000

    print("    Splitting data in training/validation/testing...")
    # Split it as 90/5/5
    train_data = data[: -2 * num_test_chars]
    valid_data = data[-2 * num_test_chars: -num_test_chars]
    test_data = data[-num_test_chars:]

    print("    Getting vocabulary...")
    vocab = get_vocabulary(train_data, vocabulary_size)

    print("    Formatting training data...")
    x_train, y_train = split_in_sequences(train_data, vocab, window_size, step_size)
    print("    Formatting validation data...")
    x_valid, y_valid = split_in_sequences(valid_data, vocab, window_size, step_size)
    print("    Formatting testing data...")
    x_test, y_test = split_in_sequences(test_data, vocab, window_size, step_size)

    with open(f'data/{name}.pickle', 'wb') as f:
        pickle.dump([x_train, y_train, x_valid, y_valid, x_test, y_test, vocabulary_size, window_size, step_size], f)


def prepare_pennchar(vocabulary_size, window_size, step_size):
    print("  Preparing Penn Treebank (PTB)...")

    with open('data/pennchar/train.txt', 'r') as f:
        train_data = f.read().split()
    with open('data/pennchar/valid.txt', 'r') as f:
        valid_data = f.read().split()
    with open('data/pennchar/test.txt', 'r') as f:
        test_data = f.read().split()

    total_length = len(train_data) + len(valid_data) + len(test_data)
    print("    Length of PTB character data set is {} with split: {} / {} / {}"
          .format(total_length, len(train_data), len(valid_data), len(test_data)))

    print("    Getting vocabulary...")
    vocab = get_vocabulary(train_data, vocabulary_size=vocabulary_size)

    print("    Formatting training data...")
    x_train, y_train = split_in_sequences(train_data, vocab, window_size, step_size)
    print("    Formatting validation data...")
    x_valid, y_valid = split_in_sequences(valid_data, vocab, window_size, step_size)
    print("    Formatting testing data...")
    x_test, y_test = split_in_sequences(test_data, vocab, window_size, step_size)

    with open(f'data/pennchar.pickle', 'wb') as f:
        pickle.dump([x_train, y_train, x_valid, y_valid, x_test, y_test, vocabulary_size, window_size, step_size], f)


def prepare_penn(vocabulary_size, window_size, step_size=1):
    print("  Preparing Penn Treebank (PTB)...")

    with open('data/penn/train.txt', 'r') as f:
        train_data = f.read().split()
    with open('data/penn/valid.txt', 'r') as f:
        valid_data = f.read().split()
    with open('data/penn/test.txt', 'r') as f:
        test_data = f.read().split()

    # Transform the data integer format and get the necessary tools to deal with that later
    train_data, word_to_index, index_to_word, t = process(train_data, vocabulary_size)
    valid_data = t.texts_to_sequences(valid_data)
    test_data = t.texts_to_sequences(test_data)

    print("    Formatting training data...")
    x_train, y_train = split_in_sequences_word_v(train_data, window_size, step_size)
    print("    Formatting validation data...")
    x_valid, y_valid = split_in_sequences_word_v(valid_data, window_size, step_size)
    print("    Formatting testing data...")
    x_test, y_test = split_in_sequences_word_v(test_data, window_size, step_size)

    with open(f'data/penn.pickle', 'wb') as f:
        pickle.dump([x_train, y_train, x_valid, y_valid, x_test, y_test, vocabulary_size, window_size, step_size], f)


def split_in_sequences_word_v(data, window_size, step_size):
    print("    Splitting the sequence in smaller sequences...")
    data = [item for sublist in data for item in sublist]
    x = []
    y = []
    for i in range(0, len(data) - window_size, step_size):  # (offset, len(text) - win + 1, step)
        inputs = data[i:i + window_size]
        label = data[i + window_size]
        x.append(inputs)
        y.append(label)

    x, y = shuffle(x, y)

    return x, y


def process(raw_text, vocabulary_size):
    # We might want to delete some words - maybe delete punctuation marks and words like "I, and". nltk library can help

    # Initialize the Tokenizer
    t = Tokenizer(num_words=vocabulary_size, oov_token=UNKNOWN)

    # Fit the tokenizer on text
    t.fit_on_texts(raw_text)

    # Turn the input into sequences of integers
    data = t.texts_to_sequences(raw_text)

    word_to_index = t.word_index
    index_to_word = {i: word for word, i in word_to_index.items()}

    return data, word_to_index, index_to_word, t


def get_vocabulary(data, vocabulary_size):  # Get the character vocabulary from the given data
    vocab = {}
    print("    Getting the vocabulary (an integer for each character)...")
    for char in data:
        vocab[str(char)] = vocab[str(char)] + 1 if char in vocab else 1

    sort = sorted(vocab, key=vocab.get, reverse=True)
    sort = [PADDING, UNKNOWN] + sort

    # for index, value in enumerate(sort):
    #    print(index, " -> ", value)

    # print(f"Full vocabulary size is {len(sort)}")

    sort = sort[:vocabulary_size]

    # print(f"Cropped vocabulary to size of {len(sort)}")

    return {value: index for index, value in enumerate(sort)}


def split_in_sequences(data, vocab, win_size, step_size):
    transformed_data = []  # Our data will now be an array of integers
    print("    Sequence of characters -> sequence of integers...")
    for c in data:
        transformed_data.append(vocab[str(c)] if str(c) in vocab else vocab[UNKNOWN])

    print("    Splitting the sequence in smaller sequences...")
    x = []
    y = []
    for i in range(0, len(transformed_data) - win_size, step_size):  # (offset, len(text) - win + 1, step)
        inputs = transformed_data[i:i + win_size]
        label = transformed_data[i + win_size]
        x.append(inputs)
        y.append(label)

    x, y = shuffle(x, y)

    return x, y


def load_data(name):
    # Check if data set is supported
    list_of_data_sets = ['enwik8', 'text8', 'pennchar', 'penn']

    if name not in list_of_data_sets:
        raise Exception("This code doesn't support the following data set!")

    with open(f'data/{name}.pickle', 'rb') as f:
        x_train, y_train, x_valid, y_valid, x_test, y_test, vocabulary_size, window_size, step_size = pickle.load(f)

    return x_train, y_train, x_valid, y_valid, x_test, y_test, vocabulary_size, window_size, step_size


def get_sequence_lengths(x):  # Full sequence length minus the padded zeros
    sequence_lengths = []
    for sequence in x:
        sequence = np.array(sequence)  # Without this line it returns full length for casual lists [1,1,1,0,0]->5
        sequence_length = len(sequence) - np.count_nonzero(sequence == 0)  # Count of non-padded symbols
        sequence_lengths.append(sequence_length)
    return sequence_lengths


if __name__ == '__main__':  # Main function
    print("Preparing character level data sets...")
    prepare_enwik8(vocabulary_size=207, window_size=10, step_size=1)
    prepare_text8(vocabulary_size=29, window_size=10, step_size=1)
    prepare_pennchar(vocabulary_size=51, window_size=10, step_size=1)
    print("Preparing word level data sets...")
    prepare_penn(vocabulary_size=9650, window_size=10, step_size=1)

    # To load a specific data set use:
    X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, X_TEST, Y_TEST, VOCABULARY_SIZE, WINDOW_SIZE, STEP_SIZE = load_data("enwik8")
