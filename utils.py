# In this file we put all the utility functions that all our experiments can use/share
import tensorflow as tf
import numpy as np
import math


def get_total_variables():
    trainable_variables = tf.trainable_variables()
    variables_total = 0
    for variable in trainable_variables:
        variables_total += np.product(variable.get_shape().as_list())
    return variables_total


def find_optimal_hidden_units(hidden_units, number_of_parameters, model_function):
    # Inspired from https://github.com/deepmind/lamb/blob/master/lamb/lamb_flags.py
    # They made a version that takes in a config file, which might be more useful to some people
    print(f"Searching for the largest possible hidden unit count"
          f",  which has <= {number_of_parameters} trainable parameters!")

    # If there wasn't given correct number of total parameters, then just use the given hidden units
    if number_of_parameters is None or number_of_parameters < 1:
        return hidden_units

    # If code goes this far, we don't care about the value in hidden_units variable anymore
    # , we will change it after it returns something

    def calculate_num_params_given_hidden_units(units):
        # Before we used "test_model = LMModel()" and in the end "del test_model", but it doesn't seem to help, but
        # If some time later you get some memory error, you can probably try this

        model_function(hidden_units=units)

        # Get the number of parameters in the current model
        variable_count = get_total_variables()

        tf.keras.backend.clear_session()  # This is necessary, so there isn't any excess stuff left

        return variable_count

    def is_good(hidden_count):
        m = calculate_num_params_given_hidden_units(hidden_count)
        correct = (m <= number_of_parameters)
        if m is None:
            print(f"Hidden units = {hidden_count}, number of trainable parameters = None BAD")
        elif correct:
            print(f"Hidden units = {hidden_count}, number of trainable parameters = {m} GOOD")
        else:
            print(f"Hidden units = {hidden_count}, number of trainable parameters = {m} BAD")
        return correct, m

    # Double the size until it's too large.
    previous_hidden_size = 1
    hidden_size = 1
    good, n = is_good(hidden_size)
    while good:
        previous_hidden_size = hidden_size
        hidden_size = max(hidden_size + 1, int(hidden_size * math.sqrt(1.2 * number_of_parameters / n)))
        good, n = is_good(hidden_size)

    # Find the real answer in the range – [previous_hidden_size, hidden_size] range
    def find_answer(lower, upper):
        while lower < upper - 1:  # While the difference is bigger than 1
            # The number of parameters is likely to be at least quadratic in
            # hidden_size. Find the middle point in log space.
            # math.exp does e^x, where x is given.
            # math.log does ln(x) aka e^y=x
            middle = int(math.exp((math.log(upper) + math.log(lower)) / 2))
            # The middle has to be 1 larger than the bottom limit or 1 smaller then the upper limit
            middle = min(max(middle, lower + 1), upper - 1)
            if is_good(middle)[0]:
                lower = middle
            else:
                upper = middle
        return lower

    return find_answer(previous_hidden_size, hidden_size)


def gelu(x):
    return x * tf.sigmoid(1.702 * x)


# Printing some information in similar style for all the experiments
def print_trainable_variables():
    variables_total = get_total_variables()
    # Some divide it by 1024 instead of 1000
    if variables_total >= 1000000:
        print("Learnable parameters:", variables_total / 1000 / 1000, 'M')  # , flush=True)
    else:
        print("Learnable parameters:", variables_total / 1000, 'K')  # , flush=True)
