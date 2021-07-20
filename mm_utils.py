# This file contains the main function which, when ran, processes the downloaded data and saves it in a format we want.
# This file also contains helper functions that are related to music modeling, such as loading data

import os.path
import pickle

import numpy as np

downloaded_data_path = "data/mm/unchanged/"
prepared_data_path = "data/mm/ready/"
supported_data_sets = ["JSB Chorales", "MuseData", "Nottingham", "Piano-midi.de"]


def check_if_data_sets_are_downloaded():
    """
        This function checks if the data sets are downloaded, and returns an error if they are not.
    """

    # If the downloaded data path doesn't exist, raise an error
    if not os.path.exists(downloaded_data_path):
        raise FileNotFoundError("The downloaded data path you specified doesn't exist!")

    # Go through each of the supported data sets
    for data_set_name in supported_data_sets:
        # If the data set doesn't exist in the folder, raise an error
        if not os.path.exists(downloaded_data_path + data_set_name + ".pickle"):
            raise FileNotFoundError(f"There is no {data_set_name}.pickle in the downloaded data folder!")

    print("All the supported data sets have been found in your specified folder!")


def smallest(a, b):
    """
        This function returns the smallest of the numbers (but it does checks for None).

        Input:
            a: int, first number;
            b: int, second number.

        Output:
            smallest: int, the smallest number of the passed numbers (the non-None number, if one of the number was
                None, and None, if both numbers were None).
    """

    if a is None:
        return b
    if b is None or a < b:
        return a
    else:
        return b


def biggest(a, b):
    """
        This function returns the biggest of the numbers (but it does checks for None).

        Input:
            a: int, first number;
            b: int, second number.

        Output:
            biggest: int, the biggest number of the passed numbers (the non-None number, if one of the number was None,
                and None, if both numbers were None).
    """

    if a is None:
        return b
    if b is None or a > b:
        return a
    else:
        return b


def analyse_data(data_path=downloaded_data_path, mode="full"):
    """
        This function analyses the data sets in the passed data path. We use this to understand the format the data sets
            are in.

        Input:
            data_path: string, the path to the data set, must be same as the downloaded or prepared data path
                (the paths where the music data sets are, doesn't matter prepared or not);
            mode: string, name of the mode, must be "full" or "mini" (meaning print full analysis or the short one).
    """

    # Check if mode was specified correctly
    assert mode in ["mini", "full"], "Invalid value for variable mode, it must be \"mini\" or \"full\"!"
    # Check if the data path was specified correctly
    assert data_path in [prepared_data_path, downloaded_data_path], \
        "Invalid value for variable data_path, it must be equal to prepared or downloaded data path!"

    print("Starting data analysis...")

    # Go through each of the data sets
    for data_set_name in supported_data_sets:
        print(6 * ' ', data_set_name)

        # Read the data set from the path
        with open(f'{data_path}{data_set_name}.pickle', 'rb') as f:
            # If the path is equal to the prepared path, then it has 4 values on the go
            # If the path is equal to the downloaded path, then we have to get access the values using 'train', 'valid,
            # 'test' keys
            if data_path == prepared_data_path:
                train, valid, test, vocab = pickle.load(f)
            else:
                data = pickle.load(f)
                # All 4 music modeling data sets have 3 keys - 'test', 'train', 'valid'
                train = data['train']
                valid = data['valid']
                test = data['test']

            # The data sets have 3 dimensional sequences, we name them as follows, entire data is called an album, album
            # has many songs, songs have many time steps, time steps have a certain amount of values

            # Statistics for the whole data set
            global_album_range = [None, None]
            global_song_range = [None, None]
            global_time_step_range = [None, None]
            global_value_range = [None, None]

            # Go through each part – training, validation, testing
            for item, description in [(train, "Train"), (valid, "Valid"), (test, "Test")]:
                # Statistics for a part of the data set
                album_size = len(item)
                song_range = [None, None]
                song_length_distribution = {}
                time_step_range = [None, None]
                value_range = [None, None]

                # Go through all the big sequences
                for seq in item:
                    song_length_in_hundreds = len(seq) // 100
                    if song_length_in_hundreds in song_length_distribution:
                        song_length_distribution[song_length_in_hundreds] += 1
                    else:
                        song_length_distribution[song_length_in_hundreds] = 1
                    song_range[0] = smallest(song_range[0], len(seq))
                    song_range[1] = biggest(song_range[1], len(seq))

                    for seq2 in seq:
                        time_step_range[0] = smallest(time_step_range[0], len(seq2))
                        time_step_range[1] = biggest(time_step_range[1], len(seq2))

                        for number in seq2:
                            value_range[0] = smallest(value_range[0], number)
                            value_range[1] = biggest(value_range[1], number)

                # If the full print mode is on print the statistics from the data slice, for example, training
                if mode == "full":
                    print(12 * ' ', description)  # Train, Valid, Test
                    print(18 * ' ', f"Song count is {album_size}")
                    print(18 * ' ', f"Song time steps range from {song_range[0]} to {song_range[1]}")
                    print(18 * ' ', end=" Song length distribution: | ")
                    for key in sorted(song_length_distribution.keys()):
                        print(f"[{key * 100} - {(key + 1) * 100 - 1}]: {song_length_distribution[key]}",
                              end=" | ")
                    print()
                    print(18 * ' ', f"Time step has values in range  from {time_step_range[0]} to {time_step_range[1]}")
                    print(18 * ' ', f"Value range - from {value_range[0]} to {value_range[1]}")

                # The data slice (for example, training) probably will update some of statistics for the data set, so we
                # do it here
                global_album_range[0] = smallest(global_album_range[0], album_size)
                global_album_range[1] = biggest(global_album_range[1], album_size)
                global_song_range[0] = smallest(global_song_range[0], song_range[0])
                global_song_range[1] = biggest(global_song_range[1], song_range[1])
                global_time_step_range[0] = smallest(global_time_step_range[0], time_step_range[0])
                global_time_step_range[1] = biggest(global_time_step_range[1], time_step_range[1])
                global_value_range[0] = smallest(global_value_range[0], value_range[0])
                global_value_range[1] = biggest(global_value_range[1], value_range[1])

            # Print out the statistics for the data set
            print(12 * ' ', f"Total")
            print(18 * ' ', f"Song count ranges from {global_album_range[0]} to {global_album_range[1]}")
            print(18 * ' ', f"Song time steps range from {global_song_range[0]} to {global_song_range[1]}")
            print(18 * ' ',
                  f"Time step has values in range from {global_time_step_range[0]} to {global_time_step_range[1]}")
            print(18 * ' ', f"Values range - from {global_value_range[0]} to {global_value_range[1]}")


# Prepares the data set with passed name
def prepare_data(name):
    """
        This function prepares the requested data set.

        Input:
            name: string, name of the data set.
    """

    # Check if data set is supported
    if name not in supported_data_sets:
        raise Exception("This code doesn't support the following data set!")

    print(f"Preparing {name} data set...")

    # Read the data set .pickle file
    with open(f'{downloaded_data_path}{name}.pickle', 'rb') as f:
        data = pickle.load(f)
        # All 4 music data sets have 3 keys - 'test', 'train', 'valid', we access them to get the values
        train = data['train']
        valid = data['valid']
        test = data['test']

    # We mask all 3 data slices
    train = mask_the_data(train)
    valid = mask_the_data(valid)
    test = mask_the_data(test)

    # We save the data set in the prepared data folder
    with open(f'{prepared_data_path}{name}.pickle', 'wb') as f:
        pickle.dump([train, valid, test, MIDI_numbers], f)


# Masks data in a binary format
def mask_the_data(data):
    """
        This function masks the music modeling data.

        Input:
            data: list, 3 dimensional data, the last dimension is, for example, [31, 32, 77].

        Output:
            masked_data: list, passed data, but it has been masked data (3 dimensional, the last dimension is,
                for example, [0, 0, 0, ..., 1, 0].)
    """

    # This will hold the entire masked data
    masked_data = []

    # Go through all the "songs"
    for seq in data:
        song = []

        # Go through all the time steps (sounds at a certain time) in a single song
        for seq2 in seq:
            # Here each seq2 is [1, 2, 3] ((1, 2, 3) for JSB Chorales)

            if not seq2:  # If this time step is empty, we skip it
                continue

            # Some time steps sequences are tuples (the ones in JSB Chorales data set), so we change them to a list
            seq2 = list(seq2)

            # Mask the time step
            masked_sequence = binary_mask(seq2, MIDI_numbers, MIDI_delay)

            # Add the time step to the "song"
            song.append(masked_sequence)
        # Add the song to the "album"
        masked_data.append(song)

    return masked_data


def binary_mask(numbers, mask_size, delay=0):
    """
        This function binary masks the passed list of numbers.

        Input:
            numbers: list, a list of numbers, for example, [1, 2, 3];
            mask_size: int, total numbers possible in this mask, for example, for MIDI numbers – 88;
            delay: int, number, with which the list of numbers start, for example, for MIDI numbers – 21.

        Output:
            mask: list, numbers masked in a binary mask.

        Example:
            Input:
                numbers = [1, 2, 3]
                mask_size = 8
                delay = 1
            Output:
                mask = [1, 1, 1, 0, 0, 0, 0, 0]
    """

    # Subtract delay from all the numbers
    numbers = [x - delay for x in numbers]

    # Create an array of length [mask_size], filled with zeros
    mask = np.zeros(mask_size)

    # Go through each of the numbers and put a one in number'th place in the mask array
    for number in numbers:
        mask[number] = 1

    return mask


if __name__ == '__main__':  # Main function
    # We check if the data sets are downloaded, and raise an error if they are not
    check_if_data_sets_are_downloaded()

    # Print out the analysis of the unprepared data
    analyse_data(data_path=downloaded_data_path, mode="full")

    # The data sets are in MIDI note numbers, which are from 21 and 108, both points included
    MIDI_numbers = 108 - 21 + 1  # So 88 different numbers
    MIDI_delay = 21

    print("Preparing music data sets...")

    # We prepare each supported data set
    for data_set in supported_data_sets:
        prepare_data(data_set)

    # Print out the analysis of the prepared data
    analyse_data(data_path=prepared_data_path, mode="full")


def load_data(name):
    """
        This function loads the requested data set.

        Input:
            name: string, name of the data set.

        Output:
            train_data: list, a list of songs for training;
            valid_data: list, a list of songs for validation;
            test_data: list, a list of songs for testing;
            vocabulary_size: int, size of the vocabulary in the data set
    """

    print("Started loading data...")

    # Check if data set is supported
    if name not in supported_data_sets:
        raise Exception("This code doesn't support the following data set!")

    # Read the data set from a .pickle file
    with open(f'{prepared_data_path}{name}.pickle', 'rb') as f:
        train_data, valid_data, test_data, vocabulary_size = pickle.load(f)

    return train_data, valid_data, test_data, vocabulary_size


# Splits the data set in parts as the parameters passed declare
def split_data_in_parts(data, window_size, step_size, time_step_feature_count, min_length=1):
    """
        This function splits the passed data in sequences. Later the model will try to predict these sequences shifted
            by one.

        Input:
            data: list, a list of songs / sequences;
            window_size: int, size of the window;
            step_size: size of the step we make, when taking sequences form the larger sequences;
            time_step_feature_count: int, length of the time step dimension;
            min_length: int, minimum length of the sequence (without the padding).

        Output:
            sequences: list, a list of input sequences, that are ready to be fed in the neural network;
            targets: list, a list of output sequences, that are ready to be fed in the neural network;
            sequence_lengths: list, a list of sequence lengths.
    """

    # Create the variables in which we will hold the split data
    sequences = []
    targets = []
    sequence_lengths = []

    # This will bet the padding in case if a sample is shorter than window_size
    time_step_zeros = np.zeros(time_step_feature_count)

    # Go through each "song" / sequence in the data
    for sequence in data:
        # We find the length of the sequence
        sequence_length = len(sequence)

        # So we later know how much data has remained
        last_index = 0
        # We go over the sequence to find all the full length sequences we can find, and we add them to their
        # corresponding variables
        for i in range(0, sequence_length - window_size, step_size):
            # Add first window_size data points from the index to the sequences variable
            sequences.append(sequence[i:i + window_size])

            # Add first window_size data points from the [index + 1] to the targets variable
            targets.append(sequence[i + 1:i + window_size + 1])

            # Add the full sequence length, that is window size, to the sequence lengths variable
            sequence_lengths.append(window_size)

            # We set the last index to next untouched data
            last_index = i + step_size

        # We must add the remaining length data by padding it
        remaining_data = sequence[last_index:]
        remaining_length = len(remaining_data)

        # If the data remaining is longer than the minimum length plus one, then we will also add it to the return
        # sequences
        if remaining_length >= min_length + 1:
            # Take the remaining data
            new_sequence = remaining_data[0:remaining_length - 1]
            new_target = remaining_data[1:remaining_length]

            # Pad the sequence until it reach full window length
            for i in range(window_size - len(new_sequence)):
                # Add a time step full of zeros
                new_sequence.append(time_step_zeros)
                new_target.append(time_step_zeros)

            # Add the input and output sequence and sequence length to their corresponding variables
            sequences.append(new_sequence)
            targets.append(new_target)
            sequence_lengths.append(remaining_length - 1)

    return sequences, targets, sequence_lengths
