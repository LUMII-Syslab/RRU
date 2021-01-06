# Importing numpy so we can use numpy arrays
import numpy as np
# Importing pickle so we can read and save pickle files
import pickle
# Importing zipfile so we can open zip files
import zipfile

# Importing Tokenizer so we can transform words to numbers easily
from tensorflow.keras.preprocessing.text import Tokenizer

supported_data_sets = ["enwik8", "text8", "pennchar", "penn"]
unchanged_data_path = "data/lm/unchanged/"
prepared_data_path = "data/lm/ready/"

PADDING = "C_PAD"
UNKNOWN = "C_UNK"


# Prepares enwik8 or text8 data set (depending on which of their names was passed)
def prepare_enwik8_or_text8(name, vocabulary_size):
    print(f"  Preparing {name}...")
    # Read data set to a variable
    if name == "enwik8":
        data = zipfile.ZipFile(f"{unchanged_data_path}enwik8.zip").read(name)
    elif name == "text8":
        data = zipfile.ZipFile(f"{unchanged_data_path}text8.zip").read(name)
    else:
        raise Exception("    This function doesn't allow data sets, that aren't enwik8 or text8")

    print(f"    Length of {name} data set is {len(data)}")

    # Standard data set split for enwik8 and text8 is 90%/5%/5%
    validation_split_percentage = 5
    test_split_percentage = 5
    num_validation_chars = len(data) * validation_split_percentage // 100
    num_test_chars = len(data) * test_split_percentage // 100
    num_validation_and_test_chars = num_validation_chars + num_test_chars

    print("    Splitting data in training/validation/testing...")
    train_data = data[: -num_validation_and_test_chars]
    valid_data = data[-num_validation_and_test_chars: -num_test_chars]
    test_data = data[-num_test_chars:]

    print("    Getting vocabulary...")
    vocabulary = get_vocabulary(train_data, vocabulary_size)

    train_data = numerate(train_data, vocabulary)
    valid_data = numerate(valid_data, vocabulary)
    test_data = numerate(test_data, vocabulary)

    with open(f'{prepared_data_path}{name}.pickle', 'wb') as f:
        pickle.dump([train_data, valid_data, test_data, vocabulary_size], f)


# Prepares PTB character-level data set
def prepare_pennchar(vocabulary_size):
    print("  Preparing Penn Treebank (PTB)...")

    with open(f'{unchanged_data_path}pennchar/train.txt', 'r') as f:
        train_data = f.read().split()
    with open(f'{unchanged_data_path}pennchar/valid.txt', 'r') as f:
        valid_data = f.read().split()
    with open(f'{unchanged_data_path}pennchar/test.txt', 'r') as f:
        test_data = f.read().split()

    total_length = len(train_data) + len(valid_data) + len(test_data)
    print(f"    Length of PTB character data set is {total_length} with split: {len(train_data)} / {len(valid_data)} / {len(test_data)}")

    print("    Getting vocabulary...")
    vocabulary = get_vocabulary(train_data, vocabulary_size=vocabulary_size)

    train_data = numerate(train_data, vocabulary)
    valid_data = numerate(valid_data, vocabulary)
    test_data = numerate(test_data, vocabulary)

    with open(f'{prepared_data_path}pennchar.pickle', 'wb') as f:
        pickle.dump([train_data, valid_data, test_data, vocabulary_size], f)


# Gets the character vocabulary from the given data
def get_vocabulary(data, vocabulary_size):
    vocabulary = {}
    print("    Getting the vocabulary (an integer for each character)...")
    for char in data:
        vocabulary[str(char)] = vocabulary[str(char)] + 1 if char in vocabulary else 1

    sort = sorted(vocabulary, key=vocabulary.get, reverse=True)
    sort = [PADDING, UNKNOWN] + sort

    # for index, value in enumerate(sort):
    #    print(index, " -> ", value)

    # print(f"Full vocabulary size is {len(sort)}")

    sort = sort[:vocabulary_size]

    # print(f"Cropped vocabulary to size of {len(sort)}")

    return {value: index for index, value in enumerate(sort)}


# Numerates the data by the passed vocabulary
def numerate(data, vocab):
    transformed_data = []
    print("    Sequence of characters -> sequence of integers...")
    for c in data:
        transformed_data.append(vocab[str(c)] if str(c) in vocab else vocab[UNKNOWN])
    return transformed_data


# Prepares PTB word-level data set
def prepare_penn(vocabulary_size):
    print("  Preparing Penn Treebank (PTB)...")

    with open(f'{unchanged_data_path}penn/train.txt', 'r') as f:
        train_data = f.read().split()
    with open(f'{unchanged_data_path}penn/valid.txt', 'r') as f:
        valid_data = f.read().split()
    with open(f'{unchanged_data_path}penn/test.txt', 'r') as f:
        test_data = f.read().split()

    # Transform the data integer format and get the necessary tools to deal with that later
    train_data, word_to_index, index_to_word, t = process(train_data, vocabulary_size)
    valid_data = t.texts_to_sequences(valid_data)
    test_data = t.texts_to_sequences(test_data)

    train_data = [item for sublist in train_data for item in sublist]
    valid_data = [item for sublist in valid_data for item in sublist]
    test_data = [item for sublist in test_data for item in sublist]

    with open(f'{prepared_data_path}penn.pickle', 'wb') as f:
        pickle.dump([train_data, valid_data, test_data, vocabulary_size], f)


# Turns the passed data into numbers, and returns dictionaries and tokenizer to transform other data
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


# Loads the data set with the passed name
def load_data(name):
    print("Started loading data...")
    # Check if data set is supported
    if name not in supported_data_sets:
        raise Exception("This code doesn't support the following data set!")

    with open(f'{prepared_data_path}{name}.pickle', 'rb') as f:
        train_data, valid_data, test_data, vocabulary_size = pickle.load(f)

    return train_data, valid_data, test_data, vocabulary_size


# Get the indexes from which the training samples will start
def get_window_indexes(data_length, window_size, step_size):
    indexes = []
    for i in range(0, data_length - window_size, step_size):
        indexes.append(i)

    return indexes


# Get data from indexes (We did so we could save on memory)
def get_input_data_from_indexes(data, indexes, window_size):
    sequences = []
    targets = []
    for i in indexes:
        sequences.append(data[i:i + window_size])
        targets.append(data[i + 1:i + window_size + 1])

    return sequences, targets


if __name__ == '__main__':  # Main function
    print("Preparing character-level data sets...")
    prepare_enwik8_or_text8(name="enwik8", vocabulary_size=207)
    prepare_enwik8_or_text8(name="text8", vocabulary_size=29)
    prepare_pennchar(vocabulary_size=51)
    print("Preparing word-level data sets...")
    prepare_penn(vocabulary_size=9650)

    # To load a specific data set use:
    TRAIN_DATA, VALID_DATA, TEST_DATA, VOCABULARY_SIZE = load_data("enwik8")


# Returns a state with zeros with the correct shape
def get_zeros_state(number_of_layers, batch_size, hidden_units, state_is_tuple=False):
    # state = np.zeros((number_of_layers, len(x_batch), hidden_units))
    zero_state = np.zeros((batch_size, hidden_units))
    if state_is_tuple:
        zero_state = [zero_state, zero_state]
    state = []
    for j in range(number_of_layers):
        state.append(zero_state)
    return state
