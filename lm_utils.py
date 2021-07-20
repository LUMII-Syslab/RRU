# This file contains the main function which, when ran, processes the downloaded data and saves it in a format we want.
# This file also contains helper functions that are related to language modeling, such as loading data

import pickle
import zipfile

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

supported_data_sets = ["enwik8", "text8", "pennchar", "penn"]
unchanged_data_path = "data/lm/unchanged/"
prepared_data_path = "data/lm/ready/"

# Strings used for padding and unknown symbols
PADDING = "C_PAD"
UNKNOWN = "C_UNK"


# Prepares enwik8 or text8 data set (depending on which of their names was passed)
def prepare_wikipedia_data_set(name, vocabulary_size):
    """
        This function prepares the requested Wikipedia dump data set.

        Input:
            name: string, the name of the data set you want prepared. Possibilities are "enwik8" and "text8";
            vocabulary_size: int, size of the vocabulary you want the data set to have.
    """

    print(f"  Preparing {name}...")

    # If the name of the data set is "enwik8" or "text8", read it to a variable
    if name == "enwik8":
        data = zipfile.ZipFile(f"{unchanged_data_path}enwik8.zip").read(name)
    elif name == "text8":
        data = zipfile.ZipFile(f"{unchanged_data_path}text8.zip").read(name)
    else:
        raise Exception("    This function doesn't allow data sets, that aren't enwik8 or text8")

    print(f"    Length of {name} data set is {len(data)}")

    # Standard data set split (the one most of the publications use) for enwik8 and text8 is 90%/5%/5%
    validation_split_percentage = 5
    test_split_percentage = 5
    num_validation_chars = len(data) * validation_split_percentage // 100
    num_test_chars = len(data) * test_split_percentage // 100
    num_validation_and_test_chars = num_validation_chars + num_test_chars

    # Split the data set in 3 parts – training, validation, testing
    print("    Splitting data in training/validation/testing...")
    train_data = data[: -num_validation_and_test_chars]
    valid_data = data[-num_validation_and_test_chars: -num_test_chars]
    test_data = data[-num_test_chars:]

    # Get the vocabulary of the training data
    print("    Getting vocabulary...")
    vocabulary = get_vocabulary(train_data, vocabulary_size)

    # Numerate the data based on the vocabulary you just got
    train_data = numerate(train_data, vocabulary)
    valid_data = numerate(valid_data, vocabulary)
    test_data = numerate(test_data, vocabulary)

    # Save the data set in a .pickle file, in the prepared data set folder
    with open(f'{prepared_data_path}{name}.pickle', 'wb') as f:
        pickle.dump([train_data, valid_data, test_data, vocabulary_size], f)


def prepare_pennchar(vocabulary_size):
    """
        This function prepares the PTB character-level data set.

        Input:
            vocabulary_size: int, size of the vocabulary you want the data set to have.
    """

    print("  Preparing Penn Treebank (PTB) character-level...")

    # Reading all data set's files
    with open(f'{unchanged_data_path}pennchar/train.txt', 'r') as f:
        train_data = f.read().split()
    with open(f'{unchanged_data_path}pennchar/valid.txt', 'r') as f:
        valid_data = f.read().split()
    with open(f'{unchanged_data_path}pennchar/test.txt', 'r') as f:
        test_data = f.read().split()

    # Print the length of each part of the data set
    total_length = len(train_data) + len(valid_data) + len(test_data)
    print(f"    Length of PTB character data set is {total_length} with split:"
          f" {len(train_data)} / {len(valid_data)} / {len(test_data)}")

    # Get the vocabulary of the training data
    print("    Getting vocabulary...")
    vocabulary = get_vocabulary(train_data, vocabulary_size=vocabulary_size)

    # Numerate the data based on the vocabulary you just got
    train_data = numerate(train_data, vocabulary)
    valid_data = numerate(valid_data, vocabulary)
    test_data = numerate(test_data, vocabulary)

    # Save the data set in a .pickle file, in the prepared data set folder
    with open(f'{prepared_data_path}pennchar.pickle', 'wb') as f:
        pickle.dump([train_data, valid_data, test_data, vocabulary_size], f)


# Gets the character vocabulary from the given data
def get_vocabulary(data, vocabulary_size):
    """
        This function returns the vocabulary of the passed data.

        Input:
            data: string, the data we want to get the vocabulary from;
            vocabulary_size: int, size of the vocabulary you want the data set to have.

        Output:
            vocabulary: dict, returns a dictionary that can transform the characters into numbersl
    """

    # We count the times each character appears in the data
    vocabulary = {}
    print("    Getting the vocabulary (an integer for each character)...")
    for char in data:
        vocabulary[str(char)] = vocabulary[str(char)] + 1 if char in vocabulary else 1

    # We sort the dictionary in a descending manner based on the times each character appeared in the data set
    sort = sorted(vocabulary, key=vocabulary.get, reverse=True)
    # We add the padding and unknown symbol labels to the vocabulary
    sort = [PADDING, UNKNOWN] + sort

    # for index, value in enumerate(sort):
    #    print(index, " -> ", value)

    # print(f"Full vocabulary size is {len(sort)}")

    # We cut the vocabulary so it has a size of vocabulary_size
    sort = sort[:vocabulary_size]

    # print(f"Cropped vocabulary to size of {len(sort)}")

    # Return a dictionary from which you can transform each character into a unique number
    return {value: index for index, value in enumerate(sort)}


def numerate(data, vocab):
    """
        This function transforms the data from text to numbers.

        Input:
            data: string / list, data, that we will transform into numbers;
            vocab: dict, a vocabulary that will tell us what number should we set each character as.

        Output:
            transformed_data: list, the data, that has been transformed into numbers.
    """

    # A list in which we will hold the transformed characters (numbers)
    transformed_data = []

    print("    Sequence of characters -> sequence of integers...")

    # For each character in the data
    for c in data:
        # If the character is in the vocabulary, we give it's number, if it's not, we give it the index that is used for
        # the unknown symbols
        transformed_data.append(vocab[str(c)] if str(c) in vocab else vocab[UNKNOWN])

    return transformed_data


# Prepares PTB word-level data set
def prepare_penn(vocabulary_size):
    """
        This function prepares the PTB word-level data set.

        Input:
            vocabulary_size: int, size of the vocabulary you want the data set to have.
    """

    print("  Preparing Penn Treebank (PTB) word-level...")

    # Reading all data set's files
    with open(f'{unchanged_data_path}penn/train.txt', 'r') as f:
        train_data = f.read().split()
    with open(f'{unchanged_data_path}penn/valid.txt', 'r') as f:
        valid_data = f.read().split()
    with open(f'{unchanged_data_path}penn/test.txt', 'r') as f:
        test_data = f.read().split()

    # Transform the data into an integer format and get the necessary tools to transform the other data
    train_data, word_to_index, index_to_word, t = process(train_data, vocabulary_size)

    # Transform validation and testing data into numbers, using the Tokenizer you just got
    valid_data = t.texts_to_sequences(valid_data)
    test_data = t.texts_to_sequences(test_data)

    # Currently the list is full of list with one number, that is, it is something like this [[1], [2], [3]], we want it
    # to be something like this [1, 2, 3]
    train_data = [item for sublist in train_data for item in sublist]
    valid_data = [item for sublist in valid_data for item in sublist]
    test_data = [item for sublist in test_data for item in sublist]

    # Save the data set in a .pickle file, in the prepared data set folder
    with open(f'{prepared_data_path}penn.pickle', 'wb') as f:
        pickle.dump([train_data, valid_data, test_data, vocabulary_size], f)


def process(raw_text, vocabulary_size):
    """
        This function transforms the passed data into numbers, and returns the tools with which you will be able to
            transform the rest of the data.returns the requested cell's configuration

        Input:
            raw_text: list, a list of words that need to be transformed into numbers.
            vocabulary_size: int, size of the vocabulary you want the data set to have.

        Output:
            data: list, a list of numbers (that used to be words);
            word_to_index: dict, dictionary with which you can transform words into numbers;
            index_to_word: dict, dictionary with whihc you can transform numbers into words;
            t: Tokenizer, an object which has been trained on the passed text, so you can use it to easy transform other
                data to fit the vocabulary it had.
    """

    # We later might want to delete some words - maybe delete punctuation marks and words like "I", "and" and others.
    # For example, nltk library can help us do it.

    # Initialize the Tokenizer
    t = Tokenizer(num_words=vocabulary_size, oov_token=UNKNOWN)

    # Fit the tokenizer on text
    t.fit_on_texts(raw_text)

    # Turn the input into sequences of integers
    data = t.texts_to_sequences(raw_text)

    # Get the dictionaries that will be able to transform the data in both ways
    word_to_index = t.word_index
    index_to_word = {i: word for word, i in word_to_index.items()}

    return data, word_to_index, index_to_word, t


# Loads the data set with the passed name
def load_data(name):
    """
        This function loads and returns the requested data set.

        Input:
            name: string, the name of the data set.

        Output:
            train_data: list, a list of numbers (which represent words) for training;
            valid_data: list, a list of numbers (which represent words) for validation;
            test_data: list, a list of numbers (which represent words) for testing;
            vocabulary_size: int, size of the vocabulary the data has.
    """

    print("Started loading data...")

    # Check if data set is supported
    if name not in supported_data_sets:
        raise Exception("This code doesn't support the following data set!")

    # Read in the data set from a .pickle file
    with open(f'{prepared_data_path}{name}.pickle', 'rb') as f:
        train_data, valid_data, test_data, vocabulary_size = pickle.load(f)

    return train_data, valid_data, test_data, vocabulary_size


def get_window_indexes(data_length, window_size, step_size):
    """
        This function returns the indexes from which the training samples will start (it doesn't return the sequences
            themselves yet, because we want to save some memory)requested cell's configuration.

        Input:
            data_length: int, the length of the data;
            window_size: int, the size of the window (sequence length);
            step_size: int, the size of the step (how often we take the data).

        Output:
            indexes: list, a list of integers representing indexes, from which we will later be able to get sequences.
    """

    indexes = []

    for i in range(0, data_length - window_size, step_size):
        indexes.append(i)

    return indexes


# Get data from indexes (We did so we could save on memory)
def get_input_data_from_indexes(data, indexes, window_size):
    """
        This function gets data from the indexes passed that you will be able to feed into the neural network.

        Input:
            data: list, a list of numbers (that represent characters or words);
            indexes: list, a list of numbers, from which we know where in the data we must take the sequences from;
            window_size: int, the size of the window (sequence length)

        Output:
            sequences: list, a list of input sequences;
            targets: list, a list of output sequences.
    """

    # We make variables which will hold the input and output sequences
    sequences = []
    targets = []

    # For each index we take a the first window_size symbols to get the input sequence, and we take the first
    # window_size symbols shifted right by 1 to get the output sequences (targets).
    for i in indexes:
        sequences.append(data[i:i + window_size])
        targets.append(data[i + 1:i + window_size + 1])

    return sequences, targets


if __name__ == '__main__':  # Main function
    # We prepare all of the language modeling data sets

    print("Preparing character-level data sets...")
    prepare_wikipedia_data_set(name="enwik8", vocabulary_size=207)
    prepare_wikipedia_data_set(name="text8", vocabulary_size=29)
    prepare_pennchar(vocabulary_size=51)
    print("Preparing word-level data sets...")
    prepare_penn(vocabulary_size=9650)

    # To load a specific data set use:
    # TRAIN_DATA, VALID_DATA, TEST_DATA, VOCABULARY_SIZE = load_data("enwik8")


def get_zeros_state(number_of_layers, batch_size, hidden_units, state_is_tuple=False):
    """
        This function returns a state filled with zeros with that has the correct shape.

        Input:
            number_of_layers: int, number of layers that RNN cells are put in;
            batch_size: int, size of the batch;
            hidden_units: int, amount of hidden_units in the RNN cell;
            state_is_tuple: bool, a boolean that says whether or not the state is a tuple (as is the case for LSTM).

        Output:
            state: list, three dimensional array filled with zeros (in the correct shape).
    """

    # We make the zero_state which has to be in the shape [batch_size, hidden_units]
    zero_state = np.zeros((batch_size, hidden_units))

    # If the state is tuple, it needs two states, so we put them together in a list
    if state_is_tuple:
        zero_state = [zero_state, zero_state]

    # A variable, which will hold the state we will return
    state = []

    # Now we need to get a list of these zero states, one for each layer
    for j in range(number_of_layers):
        state.append(zero_state)

    return state


def get_average_perplexity(total_loss, total_batches, character_level):
    average_loss = total_loss / total_batches

    if character_level:
        perplexity = average_loss / np.log(2)
    else:
        perplexity = np.exp(average_loss)

    return perplexity
