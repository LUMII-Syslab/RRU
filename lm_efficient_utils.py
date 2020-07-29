import numpy as np
import pickle
import zipfile

from tensorflow.keras.preprocessing.text import Tokenizer

PADDING = "C_PAD"
UNKNOWN = "C_UNK"


def prepare_enwik8(vocabulary_size):
    print("  Preparing enwik8...")
    prepare_enwik8_or_text8(name="enwik8", vocabulary_size=vocabulary_size)


def prepare_text8(vocabulary_size):
    print("  Preparing text8...")
    prepare_enwik8_or_text8(name="text8", vocabulary_size=vocabulary_size)


def prepare_enwik8_or_text8(name, vocabulary_size):
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

    train_data = numerate(train_data, vocab)
    valid_data = numerate(valid_data, vocab)
    test_data = numerate(test_data, vocab)

    with open(f'data/{name}.pickle', 'wb') as f:
        pickle.dump([train_data, valid_data, test_data, vocabulary_size], f)


def prepare_pennchar(vocabulary_size):
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

    train_data = numerate(train_data, vocab)
    valid_data = numerate(valid_data, vocab)
    test_data = numerate(test_data, vocab)

    with open(f'data/pennchar.pickle', 'wb') as f:
        pickle.dump([train_data, valid_data, test_data, vocabulary_size], f)


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


def numerate(data, vocab):
    transformed_data = []
    print("    Sequence of characters -> sequence of integers...")
    for c in data:
        transformed_data.append(vocab[str(c)] if str(c) in vocab else vocab[UNKNOWN])
    return transformed_data


def prepare_penn(vocabulary_size):
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

    train_data = [item for sublist in train_data for item in sublist]
    valid_data = [item for sublist in valid_data for item in sublist]
    test_data = [item for sublist in test_data for item in sublist]

    with open(f'data/penn.pickle', 'wb') as f:
        pickle.dump([train_data, valid_data, test_data, vocabulary_size], f)


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


def load_data(name):
    # Check if data set is supported
    list_of_data_sets = ['enwik8', 'text8', 'pennchar', 'penn']

    if name not in list_of_data_sets:
        raise Exception("This code doesn't support the following data set!")

    with open(f'data/{name}.pickle', 'rb') as f:
        train_data, valid_data, test_data, vocabulary_size = pickle.load(f)

    return train_data, valid_data, test_data, vocabulary_size


def get_sequence_lengths(x):  # Full sequence length minus the padded zeros
    sequence_lengths = []
    for sequence in x:
        sequence = np.array(sequence)  # Without this line it returns full length for casual lists [1,1,1,0,0]->5
        sequence_length = len(sequence) - np.count_nonzero(sequence == 0)  # Count of non-padded symbols
        sequence_lengths.append(sequence_length)
    return sequence_lengths


def one_hot_encode(labels, vocab_size):
    n_labels = len(labels)
    n_unique_labels = vocab_size  # len(np.unique(labels))
    one_hot_encoded = np.zeros((n_labels, n_unique_labels))  # Now it's [0 0] [0 0]
    one_hot_encoded[np.arange(n_labels), labels] = 1  # Now it's [1 0] [0 1]
    return one_hot_encoded


def get_window_indexes(data_length, window_size, step_size):
    indexes = []
    for i in range(0, data_length - window_size, step_size):
        indexes.append(i)

    return indexes


def get_input_data_from_indexes(data, indexes, window_size):
    sequences = []
    targets = []
    for i in indexes:
        sequences.append(data[i:i + window_size])
        targets.append(data[i + window_size])

    return sequences, targets


if __name__ == '__main__':  # Main function
    print("Preparing character level data sets...")
    prepare_enwik8(vocabulary_size=207)
    prepare_text8(vocabulary_size=29)
    prepare_pennchar(vocabulary_size=51)
    print("Preparing word level data sets...")
    prepare_penn(vocabulary_size=9650)

    # To load a specific data set use:
    TRAIN_DATA, VALID_DATA, TEST_DATA, VOCABULARY_SIZE = load_data("enwik8")