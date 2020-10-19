import pickle
import numpy as np

supported_data_sets = ["JSB Chorales", "MuseData", "Nottingham", "Piano-midi.de"]
unchanged_data_path = "data/unchanged/"
prepared_data_path = "data/ready/"


def smallest(a, b):
    if a is None:
        return b
    if b is None or a < b:
        return a
    else:
        return b


def biggest(a, b):
    if a is None:
        return b
    if b is None or a > b:
        return a
    else:
        return b


def analyse_data(data_path=unchanged_data_path, mode="full"):
    assert mode in ["mini", "full"], "Invalid value for variable mode, it must be \"mini\" or \"full\"!"
    print("Starting data analysis...")

    for data_set in supported_data_sets:
        print(6 * ' ', data_set)
        with open(f'{data_path}{data_set}.pickle', 'rb') as f:
            if data_path == prepared_data_path:
                train, valid, test, vocab = pickle.load(f)
            else:
                data = pickle.load(f)
                # All 4 music data sets have 3 keys – 'test', 'train', 'valid'
                train = data['train']
                valid = data['valid']
                test = data['test']

            # We'll store the statistics of the data set, so we can know what to feed the data into the model later
            global_big_sequences = [None, None]
            global_middle_sequences = [None, None]
            global_small_sequences = [None, None]
            global_value_range = [None, None]

            for item, description in [(train, "Train"), (valid, "Valid"), (test, "Test")]:
                big_sequences = len(item)
                middle_sequences = [None, None]
                small_sequences = [None, None]
                value_range = [None, None]

                for seq in item:
                    middle_sequences[0] = smallest(middle_sequences[0], len(seq))
                    middle_sequences[1] = biggest(middle_sequences[1], len(seq))

                    for seq2 in seq:
                        small_sequences[0] = smallest(small_sequences[0], len(seq2))
                        small_sequences[1] = biggest(small_sequences[1], len(seq2))

                        for number in seq2:
                            value_range[0] = smallest(value_range[0], number)
                            value_range[1] = biggest(value_range[1], number)

                if mode == "full":
                    print(12 * ' ', description)  # Train, Valid, Test
                    print(18 * ' ', f"Big sequences – {big_sequences}")
                    print(18 * ' ', f"Middle sequences – from {middle_sequences[0]} to {middle_sequences[1]}")
                    print(18 * ' ', f"Small sequences – from {small_sequences[0]} to {small_sequences[1]}")
                    print(18 * ' ', f"Range – from {value_range[0]} to {value_range[1]}")

                global_big_sequences[0] = smallest(global_big_sequences[0], big_sequences)
                global_big_sequences[1] = biggest(global_big_sequences[1], big_sequences)
                global_middle_sequences[0] = smallest(global_middle_sequences[0], middle_sequences[0])
                global_middle_sequences[1] = biggest(global_middle_sequences[1], middle_sequences[1])
                global_small_sequences[0] = smallest(global_small_sequences[0], small_sequences[0])
                global_small_sequences[1] = biggest(global_small_sequences[1], small_sequences[1])
                global_value_range[0] = smallest(global_value_range[0], value_range[0])
                global_value_range[1] = biggest(global_value_range[1], value_range[1])

            print(12 * ' ', "Total")
            print(18 * ' ', f"Big sequences – from {global_big_sequences[0]} to {global_big_sequences[1]}")
            print(18 * ' ', f"Middle sequences – from {global_middle_sequences[0]} to {global_middle_sequences[1]}")
            print(18 * ' ', f"Small sequences – from {global_small_sequences[0]} to {global_small_sequences[1]}")
            print(18 * ' ', f"Range – from {global_value_range[0]} to {global_value_range[1]}")


def prepare_data(name):
    # Check if data set is supported
    if name not in supported_data_sets:
        raise Exception("This code doesn't support the following data set!")

    print(f"Preparing {name} data set...")

    with open(f'{unchanged_data_path}{name}.pickle', 'rb') as f:
        data = pickle.load(f)
        # All 4 music data sets have 3 keys – 'test', 'train', 'valid'
        train = data['train']
        valid = data['valid']
        test = data['test']

    train = mask_the_data(train)
    valid = mask_the_data(valid)
    test = mask_the_data(test)

    with open(f'{prepared_data_path}{name}.pickle', 'wb') as f:
        pickle.dump([train, valid, test, MIDI_numbers], f)


def mask_the_data(data):
    big_sequences = []
    for seq in data:
        middle_sequences = []  # This is 1 "big" sequence
        for seq2 in seq:
            # Here each seq2 is [1, 2, 3]

            if not seq2:  # If this sequence is empty – [] or ()
                continue

            # Some sequences are tuple, so we change them to a list, you can add an if statement if it
            # has some performance boost
            seq2 = list(seq2)

            masked_sequence = binary_mask(seq2, MIDI_numbers, MIDI_delay)
            middle_sequences.append(masked_sequence)
        big_sequences.append(middle_sequences)
    return big_sequences


# Idea for this function is that, if we have total possibilities [1; 8], and at some time step we have
# We have numbers [1, 2, 3], then we want to get a binary mask [1,1,1,0,0,0,0,0]
# Parameters:
#   numbers – numbers at the current time step, for example, [1, 2, 3]
#   mask_size - total numbers possible, for example, for MIDI numbers – 88
#   delay – from which number the numbers start, for example, for MIDI numbers - 21
def binary_mask(numbers, mask_size, delay=0):
    numbers = [x - delay for x in numbers]
    mask = np.zeros(mask_size)
    for number in numbers:
        mask[number] = 1
    return mask


if __name__ == '__main__':  # Main function
    # analyse_data(data_path=unchanged_data_path, mode="full")

    # The data sets are in MIDI note numbers, which are from 21 and 108, both points included
    MIDI_numbers = 108 - 21 + 1  # So 88 different numbers
    MIDI_delay = 21
    # Do we need a token for padding? Currently we don't have one
    # All possible MIDI note numbers between 21 and 108 inclusive and a token for padding
    # vocabulary_size = MIDI_numbers + 1

    print("Preparing music data sets...")

    # Data sets from http://www-etud.iro.umontreal.ca/~boulanni/icml2012
    prepare_data("JSB Chorales")  # [[( : empty-()
    prepare_data("MuseData")  # [[[ : empty-[]
    prepare_data("Nottingham")  # [[[ : empty-[]
    prepare_data("Piano-midi.de")  # [[[ : empty-[]

    analyse_data(data_path=prepared_data_path, mode="mini")


def load_data(name):
    print("Started loading data...")
    # Check if data set is supported
    if name not in supported_data_sets:
        raise Exception("This code doesn't support the following data set!")

    with open(f'{prepared_data_path}{name}.pickle', 'rb') as f:
        train_data, valid_data, test_data, vocabulary_size = pickle.load(f)

    return train_data, valid_data, test_data, vocabulary_size


def split_data_in_parts(data, window_size, step_size, time_step_feature_count, min_length=1):
    sequences = []
    targets = []
    time_step_zeros = np.zeros(time_step_feature_count)
    for sequence in data:
        sequence_length = len(sequence)
        if sequence_length <= window_size + 1:  # If we can get only 1 training sample out of this sequence
            # We add the sequence only if it's larger than the stated minimum. We do this, because some people might
            # not want 1 real sample and 199 padded zero sample.
            if sequence_length >= min_length:
                new_sequence = sequence[0:sequence_length - 1]
                new_target = sequence[1:sequence_length]
                for i in range(window_size - len(new_sequence)):
                    new_sequence.append(time_step_zeros)
                    new_target.append(time_step_zeros)
                sequences.append(new_sequence)
                targets.append(new_target)
        else:
            last_index = 0
            for i in range(0, sequence_length - window_size, step_size):
                sequences.append(sequence[i:i + window_size])
                targets.append(sequence[i + 1:i + window_size + 1])
                last_index = i + step_size
            # We must pad the remaining length, if it has some left.
            remaining_data = sequence[last_index:]
            remaining_length = len(remaining_data)
            if remaining_length >= min_length:
                new_sequence = remaining_data[0:remaining_length - 1]
                new_target = remaining_data[1:remaining_length]
                for i in range(window_size - len(new_sequence)):
                    new_sequence.append(time_step_zeros)
                    new_target.append(time_step_zeros)
                sequences.append(new_sequence)
                targets.append(new_target)
    return sequences, targets
