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


def find_optimal_hidden_units(hidden_units,
                              number_of_parameters,
                              model_function,
                              ensure_divisibility_with_some_powers_of_two=False):
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
            print(f"Hidden units = {hidden_count}, number of trainable parameters = {int(m)} GOOD")
        else:
            print(f"Hidden units = {hidden_count}, number of trainable parameters = {int(m)} BAD")
        return correct, m

    # Double the size until it's too large.
    previous_hidden_size = 1
    hidden_size = 1
    good, n = is_good(hidden_size)
    while good:
        previous_hidden_size = hidden_size
        hidden_size = max(hidden_size + 1, int(hidden_size * math.sqrt(1.2 * number_of_parameters / n)))
        good, n = is_good(hidden_size)

    # Find the real answer in the range - [previous_hidden_size, hidden_size] range
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

    best_hidden_size = find_answer(previous_hidden_size, hidden_size)
    print(f"Maximum hidden_size we can use is {best_hidden_size}!")

    # As we are using nvdidia graphics cards, we can make training faster if the parameters
    # divide with some powers of 2
    # https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html#choose-params
    if ensure_divisibility_with_some_powers_of_two:
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


# Printing some information in similar style for all the experiments
def print_trainable_variables():
    variables_total = get_total_variables()
    # Some divide it by 1024 instead of 1000
    if variables_total >= 1000000:
        print("Learnable parameters:", variables_total / 1000 / 1000, 'M')  # , flush=True)
    else:
        print("Learnable parameters:", variables_total / 1000, 'K')  # , flush=True)


# Self-made function to get a readable hyperopt output
def print_trials_information(hyperopt_trials, hyperopt_choices=None, reverse_hyperopt_loguniforms=None,
                             round_uniform=None, metric="Loss", reverse_sign=False):

    if round_uniform is None:
        round_uniform = []
    if reverse_hyperopt_loguniforms is None:
        reverse_hyperopt_loguniforms = {}
    if hyperopt_choices is None:
        hyperopt_choices = {}

    def print_trial_information(single_trial):
        # keys(['state', 'tid', 'spec', 'result', 'misc', 'exp_key', 'owner', 'version', 'book_time', 'refresh_time'])
        print(f"Trial {single_trial['tid'] + 1}")
        for variable in single_trial['misc']['vals']:
            value = single_trial['misc']['vals'][variable][0]

            if variable in hyperopt_choices.keys():
                value = hyperopt_choices[variable][value]
            elif variable in reverse_hyperopt_loguniforms.keys():
                value = reverse_hyperopt_loguniforms[variable][1] - (value - reverse_hyperopt_loguniforms[variable][0])
            elif variable in round_uniform:
                value = round(value)

            print(f"    {variable} = {value}")
        if reverse_sign:
            print(f"    {metric} - {- single_trial['result']['loss']}")
        else:
            print(f"    {metric} - {single_trial['result']['loss']}")

    for trial in hyperopt_trials.trials:
        print_trial_information(trial)

    best_trial = hyperopt_trials.best_trial
    print(f"\nBest Trial")
    print_trial_information(best_trial)


def get_batch(data, current_batch_index, batch_size, fixed_batch_size=True, continuous_batches=False):
    number_of_batches = len(data) // batch_size

    if continuous_batches:
        batch = []
        if fixed_batch_size:
            for j in range(current_batch_index, number_of_batches * batch_size, number_of_batches):
                batch.append(data[j])
        else:
            for j in range(current_batch_index, len(data), number_of_batches):
                batch.append(data[j])
    else:
        # The batch_size is fixed or it's not the last
        if fixed_batch_size or current_batch_index != number_of_batches - 1:
            batch = data[current_batch_index * batch_size: current_batch_index * batch_size + batch_size]
        else:
            # Run the remaining sequences (that might not be exactly batch_size (they might be larger))
            batch = data[current_batch_index * batch_size:]

    return batch


def save_model(sess, ckpt_path, model_name):
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, ckpt_path + model_name + ".ckpt")


def restore_model(sess, ckpt_path):
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    saver = tf.compat.v1.train.Saver()
    # If there is a correct checkpoint at the path restore it
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)


class NetworkPrint:

    @staticmethod
    def training_start():
        print("|*|*|*|*|*| Starting training... |*|*|*|*|*|")

    @staticmethod
    def validation_start(epoch):
        print(f"------ Starting validation for epoch {epoch}... ------")

    @staticmethod
    def testing_start():
        print("|*|*|*|*|*| Starting testing... |*|*|*|*|*|")

    @staticmethod
    def evaluation_end(mode, metrics, time):
        print(f"Final {mode} stats | ", end='')
        for metric in metrics:
            print(f"{metric[0]}: {metric[1]}, ", end='')
        print(f"Time spent: {time}")

    @staticmethod
    def epoch_start(from_epoch, to_epoch):
        print(f"------ Epoch {from_epoch} out of {to_epoch} ------")

    @staticmethod
    def epoch_end(epoch, metrics, time):
        print(f"   Epoch {epoch} | ", end='')
        for metric in metrics:
            print(f"{metric[0]}: {metric[1]}, ", end='')
        print(f"Time spent: {time}")

    @staticmethod
    def step_results(from_step, to_step, metrics, time):
        print(f"Step {from_step} of {to_step} | ", end='')
        for metric in metrics:
            print(f"{metric[0]}: {metric[1]}, ", end='')
        print(f"Time from start: {time}")
