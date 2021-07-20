# This file contains all the utility functions that all implemented tasks use (share), such as printing the amount of
# variable in the model

import math

import numpy as np
import tensorflow as tf


def get_total_variables():
    """
        This function returns the number of trainable variables in the model.

        Output:
            variables_total: int, the amount of trainable variables in the model.
    """

    # Gets all the trainable variables
    trainable_variables = tf.trainable_variables()
    variables_total = 0
    for variable in trainable_variables:  # Go through each of the trainable variables
        # From the trainable variable's shape you can get it's variable count, add it to the total
        variables_total += np.product(variable.get_shape().as_list())
    return variables_total


def find_optimal_hidden_units(hidden_units,
                              number_of_parameters,
                              model_function,
                              ensure_divisibility_with_some_powers_of_two=False):
    """
        This function find the optimal hidden units to fit the given number of parameters

        Input:
            hidden_units: int, most likely a non-optimal amount of hidden_units;
            number_of_parameters: int, the maximum amount of trainable parameters allowed in the model;
            model_function: class, a class / function that allows you to create a model;
            ensure_divisibility_with_some_powers_of_two: bool, whether or not you want the hidden units to divide some
                powers of two (NVIDIA documentation states that this might give a performance boost).

        Output:
            hidden_units: int, the optimal amount hidden units (if number of paramaters was correctly specified).
    """

    print(f"Searching for the largest possible hidden unit count"
          f",  which has <= {number_of_parameters} trainable parameters!")

    # If there wasn't given a correct number of total parameters, then just return the given hidden units passed
    if number_of_parameters is None or number_of_parameters < 1:
        return hidden_units

    def calculate_num_params_given_hidden_units(units):
        """
            This function finds the amount of trainable variables, if the model used the passed hidden units.

            Input:
                units: int, amount of hidden units.

            Output:
                variable_count: int, the amount of trainable variables, if the passed hidden units are used.
        """

        # Creates the model with the passed hidden unit amount
        model_function(hidden_units=units)

        # Get the number of parameters in the current model
        variable_count = get_total_variables()

        # This is necessary, so there isn't any excess stuff (model parameters) left
        tf.keras.backend.clear_session()

        return variable_count

    def is_good(hidden_count):
        """
            This function tells if the passed amount of hidden units fit in the number of parameters.

            Input:
                hidden_count: int, amount of hidden units.

            Output:
                correct: bool, can we use this hidden units count;
                m: int, the amount of trainable variables we got, when we used the passed hidden units.
        """

        # We find out the number of parameters if we use the passed amount of hidden units
        m = calculate_num_params_given_hidden_units(hidden_count)

        # We check if the number of parameters we got fit in the maximum amount of trainable parameters
        correct = (m <= number_of_parameters)

        if m is None:  # If for some reason there weren't any correct variables
            print(f"Hidden units = {hidden_count}, number of trainable parameters = None BAD")
        elif correct:  # If the number of parameters fit
            print(f"Hidden units = {hidden_count}, number of trainable parameters = {int(m)} GOOD")
        else:  # If the number of parameters don't fit
            print(f"Hidden units = {hidden_count}, number of trainable parameters = {int(m)} BAD")

        return correct, m

    # Define variables that will hold the last two used hidden values (one of them will be correct, the other won't be)
    previous_hidden_size = 1
    hidden_size = 1

    # We check if we are allowed to use such hidden_size value
    good, n = is_good(hidden_size)

    # Increase the size of the model, until it's too large
    while good:
        previous_hidden_size = hidden_size

        # We increase the amount of hidden units, with a function that increases the hidden units faster in the
        # beginning (because the difference between the number_of_parameters and n most likely will be large at first),
        # and slower when it starts approaching the maximum number of parameters
        hidden_size = max(hidden_size + 1, int(hidden_size * math.sqrt(1.2 * number_of_parameters / n)))

        good, n = is_good(hidden_size)

    def find_answer(lower, upper):
        """
            This function finds the optimal amount of hidden units, when passed a range in which lowest value is
                allowed and the upper value isn't.

            Input:
                lower: int, amount of hidden units, that we are allowed to use;
                upper: int, amount of hidden units, that we aren't allowed to use.

            Output:
                lower: int, the optimal amount of hidden units.
        """

        while lower < upper - 1:  # While the difference between the range is bigger than 1
            # The number of parameters is likely to be at least quadratic in hidden_units. That's why we find the middle
            # point in the logarithmic space.
            # math.exp does e^x, where x is given.
            # math.log does ln(x) aka e^y = x, where x is given.

            # We find the center in logarithmic space
            middle = int(math.exp((math.log(upper) + math.log(lower)) / 2))

            # The middle has to be 1 larger than the bottom limit and 1 smaller then the upper limit
            middle = min(max(middle, lower + 1), upper - 1)

            if is_good(middle)[0]:  # If the middle point fits the number of parameters, it's the new lowest range point
                lower = middle
            else:  # If the middle point doesn't fit the number of parameters, it's the new highest range point
                upper = middle

        return lower

    # When we know the range in which hidden units must be, we call a function that will find the optimal value
    best_hidden_size = find_answer(previous_hidden_size, hidden_size)
    print(f"Maximum hidden_size we can use is {best_hidden_size}!")

    # If you are are using NVIDIA graphics cards, we might be able to make training faster if the parameters divide with
    # some powers of 2
    # Source: https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html#choose-params

    if ensure_divisibility_with_some_powers_of_two:
        # If the hidden units are in range (2; 128), it might be good if they divide with 2 without any remainder
        # If the hidden units are in range (128; X), it might be good if they divide with 8 without any remainder (where
        # X is some number larger than 128)
        if 2 < best_hidden_size < 128 and best_hidden_size % 2 != 0:
            chosen_hidden_size = best_hidden_size - 1
        elif 128 < best_hidden_size and best_hidden_size % 8 != 0:
            chosen_hidden_size = best_hidden_size - (best_hidden_size % 8)
        else:
            chosen_hidden_size = best_hidden_size
    else:
        chosen_hidden_size = best_hidden_size

    print(f"The hidden size we chose is {chosen_hidden_size}!")
    return chosen_hidden_size


def print_trainable_variables():
    """
        Printing the number of trainable variables in a more understandable way.
    """

    # We get the amount of trainable variables
    variables_total = get_total_variables()

    # We print the amount of variables in thousands and millions (some people use 1024, which doesn't really make sense)
    if variables_total >= 1000000:  # If there are at least 1 million parameters, print the parameter amount in millions
        print("Learnable parameters:", variables_total / 1000 / 1000, 'M')
    else:  # Otherwise print the parameter amount in thousands
        print("Learnable parameters:", variables_total / 1000, 'K')


def print_trials_information(hyperopt_trials, hyperopt_choices=None, reverse_hyperopt_loguniforms=None,
                             round_uniform=None, metric="Loss", reverse_sign=False):
    """
        This function prints information about the hyperoptimization in a overseeable manner.

        Input:
            hyperopt_trials: hyperopt.Trials, an object which holds information about the hyperoptimization;

            hyperopt_choices: dict, variable names and their list of choices (so we can find the value by accessing the
                list by index). For example:
                    choices = {
                        'variable': list_of_values
                    };

            reverse_hyperopt_loguniforms: dict, variable names and their range of values. For example:
                    reverse_log_uniforms = {
                        'variable': [1, 10]
                    };

            round_uniform: list, holds the variable names that need their values rounded. For example:
                    round_uniform = ['num_params', 'out_size']

            metric: string, the name of the final metric, for example, "Loss", "Accuracy", "Perplexity", "NLL", etc.;

            reverse_sign: bool, if True, we will reverse the sign of the final metric.
    """

    # PEP8 way to declare that variables are empty lists and dicts by default
    if round_uniform is None:
        round_uniform = []
    if reverse_hyperopt_loguniforms is None:
        reverse_hyperopt_loguniforms = {}
    if hyperopt_choices is None:
        hyperopt_choices = {}

    def print_trial_information(single_trial):
        """
            This function prints the information about a single trial.

            Input:
                single_trial: dict, a trial with it's information about the hyperoptimization
        """

        # single_trial is a dictionary that contains the following keys:'state', 'tid', 'spec', 'result', 'misc',
        # 'exp_key', 'owner', 'version', 'book_time', 'refresh_time'

        # Print which trial this is
        print(f"Trial {single_trial['tid'] + 1}")

        # Go through each variable that was optimized
        for variable in single_trial['misc']['vals']:
            # Take the value of the variable
            value = single_trial['misc']['vals'][variable][0]

            # If the variable is in hyperopt_choices, the value is only an index, we need to access the list of values
            # by index to get the real value
            # If the variable is in reverse_hyperopt_loguniforms, we need to reverse the logarithmic scale to get the
            # real value
            # If the variable is in round_uniform, we need to round the value to get the real value
            if variable in hyperopt_choices.keys():
                value = hyperopt_choices[variable][value]
            elif variable in reverse_hyperopt_loguniforms.keys():
                value = reverse_hyperopt_loguniforms[variable][1] - (value - reverse_hyperopt_loguniforms[variable][0])
            elif variable in round_uniform:
                value = round(value)

            # We print the name of the variable with it's value
            print(f"    {variable} = {value}")

        # We print the final metric, if reverse_sign is True, we need to print the negative value of the metric
        if reverse_sign:
            print(f"    {metric} - {- single_trial['result']['loss']}")
        else:
            print(f"    {metric} - {single_trial['result']['loss']}")

    # Print details of each trial in the optimization logs
    for trial in hyperopt_trials.trials:
        print_trial_information(trial)

    best_trial = hyperopt_trials.best_trial

    # Print details of the best trial in the optimization logs
    print(f"\nBest Trial")
    print_trial_information(best_trial)


def get_batch(data, current_batch_index, batch_size, fixed_batch_size=True, continuous_batches=False):
    """
        This function returns a batch of data with the correct size and taken from the correct place.

        Input:
            data: list, holds any type of data that we need to get a batch from;
            current_batch_index: int, which batch are we taking from this data;
            batch_size: int, how big a batch should be;
            fixed_batch_size: bool, does the batch have to be in a fixed size, if False, then the batch may be in a size
                that is in the range [batch_size, 2 * batch_size)
            continuous_batches: bool, do the batches have to be taken in a continuous manner. For example:
                    data = [1,2,3,4,5,6,7,8,9]
                    batch_size = 3
                    Then with continuous_batches off we get 3 batches - [1, 2, 3], [4, 5, 6], [7, 8, 9]
                    Then with continuous_batches on we get 3 batches - [1, 4, 7], [2, 5, 8], [3, 6, 9].

        Output:
            batch: list, a batch taken from the data.
    """

    # Get the number of batches
    number_of_batches = len(data) // batch_size

    if continuous_batches:  # If the batches are continuous we need to take it with a for loop
        batch = []
        if fixed_batch_size:
            for j in range(current_batch_index, number_of_batches * batch_size, number_of_batches):
                batch.append(data[j])
        else:  # This might return a batch of data that is larger than batch_size
            for j in range(current_batch_index, len(data), number_of_batches):
                batch.append(data[j])
    else:  # If batches don't have to be continuous
        # If this isn't the last batch, or if it is the last batch, but the batch size is fixed
        if fixed_batch_size or current_batch_index != number_of_batches - 1:
            batch = data[current_batch_index * batch_size: current_batch_index * batch_size + batch_size]
        else:
            # Take the remaining data
            batch = data[current_batch_index * batch_size:]

    return batch


# Class for printing training / validation and testing information
class NetworkPrint:

    @staticmethod
    def training_start():
        """
            This function prints that the training has started.
        """

        print("|*|*|*|*|*| Starting training... |*|*|*|*|*|")

    @staticmethod
    def validation_start(epoch):
        """
            This function prints that the training has started.

            Input:
                epoch: int, which contains the epoch's index, which validation phase just began.
        """

        print(f"------ Starting validation for epoch {epoch}... ------")

    @staticmethod
    def testing_start():
        """
            This function prints that the testing has started.
        """

        print("|*|*|*|*|*| Starting testing... |*|*|*|*|*|")

    @staticmethod
    def evaluation_end(mode, metrics, time):
        """
            This function prints that the validation or testing has ended, and prints out it's main results.

            Input:
                mode: string, what phase are we in, possible values are "validation" and "testing";
                metrics: list, a list of metrics, which we want to print;
                time: float, the time spent in training.
        """

        print(f"Final {mode} stats | ", end='')

        # Go through each metric and print it's values
        for metric in metrics:
            print(f"{metric[0]}: {metric[1]}, ", end='')

        print(f"Time spent: {time}", end='\n')

    @staticmethod
    def epoch_start(from_epoch, to_epoch):
        """
            This function prints that a training epoch has just started and prints out the epoch's index and the total
                epochs.

            Input:
                from_epoch: int, epoch, which just started;
                to_epoch: int, epochs total.
        """

        print(f"------ Epoch {from_epoch} out of {to_epoch} ------")

    @staticmethod
    def epoch_end(epoch, metrics, time):
        """
            This function prints that the a training epoch has just ended, and prints out it's main results.

            Input:
                epoch: int, epoch, which just ended;
                metrics: list, a list of metrics, which we want to print;
                time: float, the time spent in training.
        """

        print(f"   Epoch {epoch} | ", end='')

        # Go through each metric and print it's values
        for metric in metrics:
            print(f"{metric[0]}: {metric[1]}, ", end='')

        print(f"Time spent: {time}", end='\n')

    @staticmethod
    def step_results(from_step, to_step, metrics, time):
        """
            This function prints that the validation or testing has ended, and prints out it's main results.

            Input:
                from_epoch: int, step, which just started;
                to_epoch: int, steps total;
                metrics: list, a list of metrics, which we want to print;
                time: float, the time spent in training.
        """

        print(f"Step {from_step} of {to_step} | ", end='')

        # Go through each metric and print it's values
        for metric in metrics:
            print(f"{metric[0]}: {metric[1]}, ", end='')

        print(f"Time from start: {time}", end='\n')
