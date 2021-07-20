# This file implements the main functions for running the language modeling task, this includes: model creation; model
# training, model testing and the main function (that controls all the flow)

import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from hyperopt import hp, tpe, Trials, fmin
from sklearn.utils import shuffle

from RAdam import RAdamOptimizer
# This function will allow us to get information about the picked cell
from cell_registry import get_cell_information
# Importing some functions that will help us deal with the input data
from lm_utils import load_data, get_window_indexes, get_input_data_from_indexes, get_zeros_state, get_average_perplexity
# Importing some utility functions that will help us with certain tasks
from utils import find_optimal_hidden_units, print_trainable_variables, get_batch
from utils import print_trials_information, NetworkPrint

# If you have many GPUs available, you can specify which one to use here (they are indexed from 0)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# You can check if this line makes it train faster, while still training correctly
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

# Hyperparameters
# 1. Data parameters
data_set_name = "enwik8"  # string, one of these ["enwik8", "text8", "pennchar", "penn"]
window_size = 256  # int, >= 1 (if model isn't stateful or continuous_batches) else 1 or (>= 1 and % 2 == 0)
batch_size = 64  # int, >= 1
fixed_batch_size = False  # bool
shuffle_data = False  # bool
continuous_batches = True  # bool
# 2. Model parameters
cell_name = "RRU"  # string, one of these ["RRU", "GRRUA", "GRU", "LSTM", "MogrifierLSTM"]
HIDDEN_UNITS = 1024  # int, >= 1 (Probably way more than 1)
number_of_parameters = 48000000  # int, >= 1 (Probably way more than 1)
learning_rate = 0.001  # float, > 0
number_of_layers = 2  # int, >= 1
output_size = 128  # int, >= 1 (Probably way more than 1)
clip_gradients = True  # bool
clip_multiplier = 10.  # float
embedding_size = 32  # int, >= 1
stateful = True  # bool
zero_state_chance = 0.1  # float, 0 <= value <= 1
outer_dropout = 0  # float, 0 <= value <= 1
# 3. Training/testing process parameters
num_epochs = 1000000  # int, >= 1
break_epochs_no_gain = 3  # int, >= 0
do_hyperparameter_optimization = False  # bool
optimization_runs = 100  # int, >= 1
ckpt_path = 'ckpt_lm/'  # string
log_path = 'logdir_lm/'  # string
log_after_this_many_steps = 0  # integer, >= 0
print_after_this_many_steps = 1  # integer, >= 0
""" Hyperparameter descriptions
    Data parameters
        data_set_name
            Choose data set to test on
        window_size
            How many time steps we will unroll in RNN
            We do character-level 512, word-level 64
        batch_size
            How many data samples we feed in a single time
        fixed_batch_size
            Can some of the batches be with size [batch_size, 2 * batch_size) (So we don't have as much left over data)
        shuffle_data
            Should we shuffle the samples?
            We do character-level False, word-level True
        continuous_batches
            Should batches go continuously? This might give performance boost with a stateful RNN cell
            We do character-level True, word-level False
    Model parameters
        cell_name
            Name of the cell you want to test
        HIDDEN_UNITS
            Number of hidden units (This will only be used if the number_of_parameters is None or < 1)
        number_of_parameters
            Number of maximum allowed trainable parameters
        learning_rate
            With what learning rate we optimize the model
        number_of_layers
            How many RNN layers should we have
        output_size
            What should be the output size for the cells that have a separate output_size?
        clip_gradients
            Should we clip the gradients?
        clip_multiplier
            If clip_gradients = True, with what multiplier should we clip them?
        embedding_size
            What should be the embedding size for the data (how many dimensions)?
            We do PTB character-level 16, PTB word-level 64
        stateful
            Should the RNN cell be stateful
            We do character-level True, word-level False
        zero_state_chance
            If stateful = True, you can choose a chance that zero_state will be passed instead of last state
        outer_dropout
            What dropout should we apply over the RNN output
    Training/testing process parameters
        num_epochs
            How many epochs should we run?
        break_epochs_no_gain
            After how many epochs with no performance gain should we early stop? (0 disabled)
        do_hyperparameter_optimization
            Should we do hyperparameter optimization?
        optimization_runs
            How many runs should we run hyperparameter optimization
        ckpt_path
            Path, where we will save the model for further evaluating
        log_path
            Path, where we will store ours logs
        log_after_this_many_steps
            After how many steps should we send the data to TensorBoard (0 - don't log after any amount of steps)
        print_after_this_many_steps
            After how many steps should we print the results of training/validation/testing (0 - don't print until the last
            step)
"""

# We need to know whether the data set is character-level or word-level
character_level = True
if data_set_name in ["penn"]:
    character_level = False

# Get information about the picked cell
cell_fn, model_name, has_separate_output_size, state_is_tuple = get_cell_information(cell_name)

# If the picked cell doesn't have a separate output size, we set is as None, so we can later process that
if not has_separate_output_size:
    output_size = None

# Calculating the path, in which we will store the logs
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")  # We put current time in the name, so it is unique in each run
output_path = log_path + model_name + f'/{data_set_name}/' + current_time

# Don't print TensorFlow messages, that we don't need
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Things we will later remove
dropout_rate = 0.7
# RRU
relu_layers = 2
middle_layer_size_multiplier = 4
# LSTM
forget_bias = 1.0
# Mogrifier LSTM
feature_mask_rounds = 6
feature_mask_rank = 79


# Class for solving language modeling tasks. You can create a model, train it and test it
class LanguageModelingModel:

    def __init__(self, hidden_units):
        """
            This function (also a constructor) creates a machine learning model for solving the language modeling task.

            Input:
                hidden_units: int, the amount of hidden units to use for the RNN cell(s).
        """

        print("\nBuilding Graph...\n")

        # Build the graph
        tf.reset_default_graph()

        # Input – batch size list of integer sequences
        x = tf.placeholder(tf.int32, shape=[None, window_size], name="x")

        # Output – batch size list of integer sequences (we'll try to predict the input sequences shifted by 1)
        y = tf.placeholder(tf.int64, shape=[None, window_size], name="y")

        # Boolean value that tells us, whether or not are we in training
        training = tf.placeholder(tf.bool, name='training')

        # Outer dropout probability, so we can pass different values in training and in testing (while testing the
        # dropout should be 0.)
        outer_dropout_rate = tf.placeholder(tf.float32, name='outer_dropout_rate')

        # Instantiate our embedding matrix
        embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="word_embedding")

        # Lookup embeddings
        embed_lookup = tf.nn.embedding_lookup(embedding, x)

        # Create the RNN cell, corresponding to the one you chose above
        cells = []
        for _ in range(number_of_layers):
            if cell_name in ["RRU", "GRRUA"]:
                cell = cell_fn(hidden_units,
                               training=training,
                               output_size=output_size,
                               relu_layers=relu_layers,
                               middle_layer_size_multiplier=middle_layer_size_multiplier,
                               dropout_rate=dropout_rate)
            elif cell_name in ["LSTM"]:  # LSTM
                cell = cell_fn(hidden_units, training=training, dropout_rate=dropout_rate, forget_bias=forget_bias)
            elif cell_name in ["MogrifierLSTM"]:  # Mogrifier LSTM
                cell = cell_fn(hidden_units, training=training, dropout_rate=dropout_rate,
                               feature_mask_rank=feature_mask_rank, feature_mask_rounds=feature_mask_rounds)
            else:  # GRU
                cell = cell_fn(hidden_units, training=training, dropout_rate=dropout_rate)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        # Some cells use output size that differs from the hidden_size. And if the cell doesn't use a different
        # output_size, we need to make it as hidden units for the program to work.
        final_size = output_size
        if final_size is None:
            final_size = hidden_units

        # Extract the batch size - this allows for variable batch size
        current_batch_size = tf.shape(x)[0]

        if stateful:  # If you wanted the model to be stateful, it will need a placeholder to pass the state
            if state_is_tuple:  # LSTMs and such (they need special treatment because they have 2 states)
                initial_state = tf.placeholder(tf.float32,
                                               shape=[number_of_layers, 2, None, hidden_units],
                                               name="initial_state")
                initial_state = tf.unstack(initial_state, axis=0)
                initial_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(initial_state[idx][0], initial_state[idx][1])
                     for idx in range(number_of_layers)]
                )
            else:  # Regular shaped RRN cells
                initial_state = tf.placeholder(tf.float32,
                                               shape=[number_of_layers, None, hidden_units],
                                               name="initial_state")
                initial_state = tuple(tf.unstack(initial_state, axis=0))
        else:  # If you didn't, use the cell's inner zero state
            initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)

        # Value will have all the outputs.
        # State contains the final state.
        value, state = tf.nn.dynamic_rnn(cell,
                                         embed_lookup,
                                         initial_state=initial_state,
                                         dtype=tf.float32)

        # Instantiate weights
        weight = tf.get_variable("output", [final_size, vocabulary_size])
        # Instantiate biases
        bias = tf.Variable(tf.constant(0.0, shape=[vocabulary_size]))

        # Optionally apply relu activation for RRU and GRRUA cells
        # value = tf.nn.relu(value)

        # Reshape the outputs from [batch_size, window_size, final_size] to [batch_size x window_size, final_size]
        last = tf.reshape(value, shape=(-1, final_size))

        # Apply the passed dropout
        last = tf.nn.dropout(last, rate=outer_dropout_rate)

        # Calculate the predictions. Final form – [batch_size x window_size, vocabulary_size]
        prediction = tf.matmul(last, weight) + bias
        # Reshape the predictions for easier further calculations
        # From [batch_size x window_size, vocabulary_size] to [batch_size, window_size, vocabulary_size]
        prediction = tf.reshape(prediction, shape=(-1, window_size, vocabulary_size))

        # Predictions for the second half (because we sometimes use half_perplexity for calculations)
        half = window_size // 2
        half_prediction = prediction[:, half:, :]  # It's shape - [batch_size, ceil(window_size / 2), vocabulary_size]

        # Accuracy for the second half
        half_y = y[:, half:]  # [batch_size, ceil(window_size / 2)]
        half_correct_prediction = tf.equal(tf.argmax(half_prediction, axis=-1), half_y)
        half_accuracy = tf.reduce_mean(tf.cast(half_correct_prediction, tf.float32))
        # Accuracy for the full length
        correct_prediction = tf.equal(tf.argmax(prediction, axis=-1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Calculate losses given prediction and labels
        # Loss only used for results (second half)
        half_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=half_prediction,
                                                                                  labels=half_y))
        # Loss for optimizing and sometimes for results (full_length)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                                                                             labels=y))

        # Printing trainable variables which have "kernel" in their name
        decay_vars = [v for v in tf.trainable_variables() if 'kernel' in v.name]
        for c in decay_vars:
            print(c)

        # Declare our optimizer, we have to check which one works better.
        optimizer = RAdamOptimizer(learning_rate=learning_rate,
                                   L2_decay=0.0,
                                   decay_vars=decay_vars,
                                   clip_gradients=clip_gradients, clip_multiplier=clip_multiplier).minimize(loss)

        # What to log to TensorBoard if a number was specified for "log_after_this_many_steps" variable
        tf.summary.scalar("half_accuracy", half_accuracy)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("loss", loss)

        # Expose symbols to class
        # Parameters, that we need public
        self.hidden_units = hidden_units
        # Placeholders
        self.x = x
        self.y = y
        self.training = training
        self.outer_dropout_rate = outer_dropout_rate
        if stateful:
            self.initial_state = initial_state
        # Information you can get from this graph
        self.state = state
        self.half_accuracy = half_accuracy
        self.accuracy = accuracy
        self.half_loss = half_loss
        self.loss = loss
        # To call the optimization step (gradient descent)
        self.optimizer = optimizer

        print_trainable_variables()

        print("\nGraph Built...\n")

    def fit(self, training_data, validation_data):
        """
            This function trains the model using the training and validation data passed.

            Input:
                training_data: list, a list of numbers to be fed in the model;
                validation_data: list, a list of numbers to be fed in the model.
        """

        tf_config = tf.ConfigProto()
        # tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with tf.Session(config=tf_config) as sess:
            # We print that the training has started
            NetworkPrint.training_start()

            # We initialize the variables
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Adding writers so we can visualize accuracy, perplexity, etc. in TensorBoard
            merged_summary = tf.summary.merge_all()
            training_writer = tf.summary.FileWriter(output_path + "/training")
            validation_writer = tf.summary.FileWriter(output_path + "/validation")
            # Adding session graph to the writer, so we can look at it, if we want, in TensorBoard
            training_writer.add_graph(sess.graph)

            global fixed_batch_size
            # If we are in stateful mode, we need fixed_batch_size, because we can't pass the previous state if it might
            # have had a different batch_size
            if stateful:
                fixed_batch_size = True

            # For stateful and continuous_batches models, the step size has to be full window_size
            # Otherwise we will use window_size // 2, so we can use the loss used in the latter part of the sequence
            # (second half), because it has more context, and therefore it might have better results.
            if stateful or continuous_batches:
                training_indexes = get_window_indexes(len(training_data), window_size, window_size)
            else:
                training_indexes = get_window_indexes(len(training_data), window_size, window_size // 2)
            # We get the indexes of the data, we use step_size = window_size // 2, because for testing we will want
            # to use the half metric results
            validation_indexes = get_window_indexes(len(validation_data), window_size, window_size // 2)

            # We calculate the number of batches
            num_training_batches = len(training_indexes) // batch_size
            num_validation_batches = len(validation_indexes) // batch_size

            # Variables that help implement the early stopping if no validation perplexity decrease is observed
            epochs_no_gain = 0
            best_validation_perplexity = None

            for epoch in range(num_epochs):
                # Print that the epoch has started
                NetworkPrint.epoch_start(epoch + 1, num_epochs)

                # If necessary, shuffle data
                if shuffle_data:
                    training_indexes = shuffle(training_indexes)

                total_training_loss = 0
                total_training_accuracy = 0

                start_time = time.time()

                state = None

                extra_tasks_for_perfection = not stateful and not continuous_batches

                for i in range(num_training_batches):
                    # Get a batch of data
                    x_batch = get_batch(training_indexes, i, batch_size, fixed_batch_size, continuous_batches)

                    # If the model is stateful, then we will pass it zero_state with a certain probability
                    if stateful and (np.random.uniform() < zero_state_chance or i == 0):
                        state = get_zeros_state(number_of_layers, len(x_batch), self.hidden_units, state_is_tuple)

                    # We get sequences from the batch, that are ready to be fed in the network
                    x_batch, y_batch = get_input_data_from_indexes(training_data, x_batch, window_size)

                    # If the model is stateful we need to pass the state as well
                    if stateful:
                        feed_dict = {
                            self.x: x_batch,
                            self.y: y_batch,
                            self.training: True,
                            self.outer_dropout_rate: outer_dropout,
                            self.initial_state: state
                        }
                    else:
                        feed_dict = {
                            self.x: x_batch,
                            self.y: y_batch,
                            self.training: True,
                            self.outer_dropout_rate: outer_dropout
                        }

                    # Do we need to fetch the full length perplexity and accuracy
                    need_full = stateful or continuous_batches or i == 0

                    # If we need to log this batch, we also add summary to the sess.run
                    if log_after_this_many_steps != 0 and i % log_after_this_many_steps == 0:
                        s, _, last_state, loss, a = sess.run([merged_summary,
                                                              self.optimizer,
                                                              self.state,
                                                              self.loss if need_full else self.half_loss,
                                                              self.accuracy if need_full else self.half_accuracy],
                                                             feed_dict=feed_dict)

                        # Adding the summary to TensorBoard
                        training_writer.add_summary(s, i + epoch * num_training_batches)
                    else:
                        _, last_state, loss, a = sess.run([self.optimizer,
                                                           self.state,
                                                           self.loss if need_full else self.half_loss,
                                                           self.accuracy if need_full else self.half_accuracy],
                                                          feed_dict=feed_dict)

                    # To have 100% data covered if we had step_size = window_size // 2 , we need to have full length
                    # values for first batch
                    if extra_tasks_for_perfection and i == 0:
                        # Because we went through 2 halves, to have fair average, we need an extra addition, but we will
                        # have to divide the total values by one more later, when calculating the average values
                        total_training_loss += loss
                        total_training_accuracy += a

                    # Setting the state as the one passed from the model (if the model is stateful, we might pass it)
                    state = last_state

                    total_training_loss += loss
                    total_training_accuracy += a

                    # Print the batch results if it's the last batch or if step printing is turned on, and this is the
                    # step to print in
                    if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0) \
                            or i == num_training_batches - 1:
                        NetworkPrint.step_results(i + 1,
                                                  num_training_batches,
                                                  [["Perplexity", get_average_perplexity(loss, 1, character_level)],
                                                   ["Accuracy", a]],
                                                  time.time() - start_time)

                # By how much we need to divide the total values to get the average values
                statistics_count = (num_training_batches + 1) if extra_tasks_for_perfection else num_training_batches

                average_training_perplexity = get_average_perplexity(total_training_loss, statistics_count,
                                                                     character_level)
                average_training_accuracy = total_training_accuracy / statistics_count

                # Print the stats gained in the epoch
                NetworkPrint.epoch_end(epoch + 1,
                                       [["Average perplexity", average_training_perplexity],
                                        ["Average accuracy", average_training_accuracy]], time.time() - start_time)

                # Add perplexity and accuracy to TensorBoard

                epoch_perplexity_summary = tf.Summary()
                epoch_perplexity_summary.value.add(tag='training_epoch_perplexity',
                                                   simple_value=average_training_perplexity)
                training_writer.add_summary(epoch_perplexity_summary, epoch + 1)

                epoch_accuracy_summary = tf.Summary()
                epoch_accuracy_summary.value.add(tag='training_epoch_accuracy', simple_value=average_training_accuracy)
                training_writer.add_summary(epoch_accuracy_summary, epoch + 1)
                training_writer.flush()

                # VALIDATION STARTS HERE

                # We print that the validation has started
                NetworkPrint.validation_start(epoch + 1)

                total_validation_loss = 0
                total_validation_accuracy = 0

                start_time = time.time()

                for i in range(num_validation_batches):
                    # Get a batch of data
                    x_batch = get_batch(validation_indexes, i, batch_size, fixed_batch_size, continuous_batches)

                    # We getsequences from the indexes, that we will be able to feed in the network
                    x_batch, y_batch = get_input_data_from_indexes(validation_data, x_batch, window_size)

                    # If the model is stateful we will not feed it the old state, because for validation and testing we
                    # will always use half metric calculations, but the model expects a state to be fed in, so we always
                    # pass it a state filled with zeros
                    if stateful:
                        state = get_zeros_state(number_of_layers, len(x_batch), self.hidden_units, state_is_tuple)

                        feed_dict = {
                            self.x: x_batch,
                            self.y: y_batch,
                            self.training: False,
                            self.outer_dropout_rate: 0.,
                            self.initial_state: state
                        }
                    else:
                        feed_dict = {
                            self.x: x_batch,
                            self.y: y_batch,
                            self.training: False,
                            self.outer_dropout_rate: 0.
                        }

                    if i == 0:  # To have 100% data covered we need to have full length values for first batch
                        loss, a = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

                        # Because we went through 2 halves, to have fair average we need * 2, but + 2 batches also
                        loss = 2 * loss
                        a = 2 * a
                    else:
                        loss, a = sess.run([self.half_loss, self.half_accuracy], feed_dict=feed_dict)

                    total_validation_loss += loss
                    total_validation_accuracy += a

                    # Print the batch results if it's the last batch or if step printing is turned on, and this is the
                    # step to print in
                    if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0) \
                            or i == num_validation_batches - 1:
                        NetworkPrint.step_results(i + 1,
                                                  num_validation_batches,
                                                  [["Average perplexity",
                                                    get_average_perplexity(total_validation_loss,
                                                                           i + 2,
                                                                           character_level)],
                                                   ["Average accuracy", total_validation_accuracy / (i + 2)]],
                                                  time.time() - start_time)

                average_validation_perplexity = get_average_perplexity(total_validation_loss,
                                                                       num_validation_batches + 1, character_level)
                average_validation_accuracy = total_validation_accuracy / (num_validation_batches + 1)

                # Print the stats gained in the evaluation phase
                NetworkPrint.evaluation_end("validation", [["Average perplexity", average_validation_perplexity],
                                                           ["Average accuracy", average_validation_accuracy]],
                                            time.time() - start_time)

                # We add the final perplexity and accuracy to TensorBoard so we don't have to dig into the console logs
                # and nohup files
                perplexity_summary = tf.Summary()
                perplexity_summary.value.add(tag=f'validation_epoch_perplexity',
                                             simple_value=average_validation_perplexity)
                validation_writer.add_summary(perplexity_summary, epoch + 1)

                accuracy_summary = tf.Summary()
                accuracy_summary.value.add(tag=f'validation_epoch_accuracy', simple_value=average_validation_accuracy)
                validation_writer.add_summary(accuracy_summary, epoch + 1)
                validation_writer.flush()

                # Training and validation for the epoch is done, check if the validation perplexity was better in this
                # epoch than in the last one
                if best_validation_perplexity is None or average_validation_perplexity < best_validation_perplexity:
                    print(f"&&& New best validation perplexity - before: {best_validation_perplexity};"
                          f" after: {average_validation_perplexity} - saving model...")

                    best_validation_perplexity = average_validation_perplexity

                    # Save the model
                    saver = tf.compat.v1.train.Saver()
                    saver.save(sess, ckpt_path + model_name + ".ckpt")

                    epochs_no_gain = 0
                elif break_epochs_no_gain >= 1:  # Validation perplexity was worse, check break_epochs_no_gain
                    epochs_no_gain += 1

                    print(f"&&& No validation perplexity decrease for {epochs_no_gain} epochs,"
                          f" breaking at {break_epochs_no_gain} epochs.")

                    # The perplexity wasn't decreasing for break_epochs_no_gain times in a row, so we need to stop the
                    # training
                    if epochs_no_gain == break_epochs_no_gain:
                        print(f"&&& Maximum epochs without validation perplexity decrease reached, breaking...")
                        return

    def evaluate(self, data):
        """
            This function tests the model with the passed data.

            Input:
                data: list, a list of numbers to be fed in the model.

            Output:
                average_perplexity: float, the average perplexity gained after evaluating that passed data.
        """

        with tf.Session() as sess:
            # We print that the testing has started
            NetworkPrint.testing_start()

            # We initialize the variables
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Restore the session
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver = tf.compat.v1.train.Saver()
            # If there is a correct checkpoint at the path restore it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # Adding a writer so we can visualize accuracy, perplexity, etc. on TensorBoard
            writer = tf.summary.FileWriter(output_path + f"/testing")

            # We get the indexes of the data, we use step_size = window_size // 2, because for testing we will want to
            # use the half metric results
            indexes = get_window_indexes(len(data), window_size, window_size // 2)

            # We calculate the number of batches
            num_batches = len(indexes) // batch_size

            total_loss = 0
            total_accuracy = 0

            start_time = time.time()

            for i in range(num_batches):
                # Get a batch of data
                x_batch = get_batch(indexes, i, batch_size, fixed_batch_size, continuous_batches)

                # We getsequences from the indexes, that we will be able to feed in the network
                x_batch, y_batch = get_input_data_from_indexes(data, x_batch, window_size)

                # If the model is stateful we will not feed it the old state, because for validation and testing we will
                # always use half metric calculations, but the model expects a state to be fed in, so we always pass it
                # a state filled with zeros
                if stateful:
                    state = get_zeros_state(number_of_layers, len(x_batch), self.hidden_units, state_is_tuple)

                    feed_dict = {
                        self.x: x_batch,
                        self.y: y_batch,
                        self.training: False,
                        self.outer_dropout_rate: 0.,
                        self.initial_state: state
                    }
                else:
                    feed_dict = {
                        self.x: x_batch,
                        self.y: y_batch,
                        self.training: False,
                        self.outer_dropout_rate: 0.
                    }

                if i == 0:  # To have 100% data covered we need to have full length values for first batch
                    loss, a = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

                    # Because we went through 2 halves, to have fair average we need * 2, but + 2 batches also
                    loss = 2 * loss
                    a = 2 * a
                else:
                    loss, a = sess.run([self.half_loss, self.half_accuracy], feed_dict=feed_dict)

                total_loss += loss
                total_accuracy += a

                # Print the batch results if it's the last batch or if step printing is turned on, and this is the step
                # to print in
                if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0) \
                        or i == num_batches - 1:
                    NetworkPrint.step_results(i + 1,
                                              num_batches,
                                              [["Average perplexity",
                                                get_average_perplexity(total_loss, i + 2, character_level)],
                                               ["Average accuracy", total_accuracy / (i + 2)]],
                                              time.time() - start_time)

            average_perplexity = get_average_perplexity(total_loss, num_batches + 1, character_level)
            average_accuracy = total_accuracy / (num_batches + 1)

            # Print the stats gained in the evaluation phase
            NetworkPrint.evaluation_end("testing", [["Average perplexity", average_perplexity],
                                                    ["Average accuracy", average_accuracy]],
                                        time.time() - start_time)

            # We add the final perplexity and accuracy to TensorBoard so we don't have to dig into the console logs and
            # nohup files
            perplexity_summary = tf.Summary()
            perplexity_summary.value.add(tag=f'testing_perplexity', simple_value=average_perplexity)
            writer.add_summary(perplexity_summary, 1)

            accuracy_summary = tf.Summary()
            accuracy_summary.value.add(tag=f'testing_accuracy', simple_value=average_accuracy)
            writer.add_summary(accuracy_summary, 1)
            writer.flush()

            # We return the average perplexity (perplexity being the main metric for the language modeling data sets)
            return average_perplexity


if __name__ == '__main__':  # Main function
    # Load the data set with the name you specified
    TRAINING_DATA, VALIDATION_DATA, TESTING_DATA, vocabulary_size = load_data(data_set_name)

    # To see how it trains in small amounts, you can uncomment this (if it's usually in a 90%/5%/5% split, as it is for
    # enwik8 and text8 data sets, it now will be in a 1%/1%/1% split)
    '''
    TRAINING_DATA = TRAINING_DATA[:len(TRAINING_DATA) // 90]
    VALIDATION_DATA = VALIDATION_DATA[:len(VALIDATION_DATA) // 5]
    TESTING_DATA = TESTING_DATA[:len(TESTING_DATA) // 5]
    '''

    # From which function / class we can get the model
    model_function = LanguageModelingModel

    if not do_hyperparameter_optimization:  # If hyperparameter optimization is off
        # Find the optimal hidden units to use without surpassing the number of parameters
        HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                 number_of_parameters=number_of_parameters,
                                                 model_function=model_function)

        # Create the model with the optimal hidden units
        MODEL = model_function(HIDDEN_UNITS)

        # Train the model (validating after each epoch)
        MODEL.fit(TRAINING_DATA, VALIDATION_DATA)

        # Test the last saved model
        MODEL.evaluate(TESTING_DATA)
    else:  # If hyperparameter optimization is on
        # rounds_choice = [5, 6]  # Mogrifier LSTM
        # We need this, so we can print the hp.choice answers normally
        choices = {
            # 'rounds': rounds_choice  # Mogrifier LSTM
        }

        # Add all hp.uniform values that need to be rounded in this list (we need this, so we later can print the values
        # rounded)
        round_uniform = ['num_params']  # Everything that's not Mogrifier LSTM
        # round_uniform = ['num_params', 'rank']  # Mogrifier LSTM

        # Define the space that will be passed into the hyperopt optimizing functions
        space = [
            # hp.choice
            # hp.choice('rounds', rounds_choice),  # Mogrifier LSTM
            # hp.uniform
            hp.uniform('num_params', 10000000, 30000000),
            hp.uniform('drop', 0., 0.8),
            hp.uniform('middle', 0.1, 8.),  # RRU
            # hp.uniform('forget', -3., 3.),  # LSTM
            # hp.uniform('rank', 40, 90),  # Mogrifier LSTM
            # hp.loguniform
            hp.loguniform('lr', np.log(0.0001), np.log(0.01))
        ]


        def objective(num_params, drop, middle, lr):  # RRU
            # def objective(num_params, drop, lr):  # GRU
            # def objective(num_params, drop, forget, lr):  # LSTM
            # def objective(rounds, num_params, drop, rank, lr):  # Mogrifier LSTM
            # The function inputs must be in the same order as they are specified in the space variable
            # This function does the same steps as the above code (when hyperparameter optimization is off), but it has
            # to set the passed variables (some of them need some additional actions) and return the metric that has to
            # be minimized while doing the hyperparameter optimization

            # We need to round some of the "hp.uniform" values
            num_params = round(num_params)
            # rank = round(rank)  # Mogrifier LSTM

            # We'll optimize these parameters (we need to set them globally, because we use global variables in some of
            # the model functions, so the code is clearer and it doesn't need too many variables in each function)
            global number_of_parameters, dropout_rate, middle_layer_size_multiplier, learning_rate  # RRU
            # global number_of_parameters, dropout_rate, learning_rate  # GRU
            # global number_of_parameters, dropout_rate, forget_bias, learning_rate  # LSTM
            # global feature_mask_rounds, number_of_parameters, dropout_rate, feature_mask_rank, learning_rate  # Mogrifier LSTM
            # feature_mask_rounds = rounds  # Mogrifier LSTM
            number_of_parameters = num_params
            dropout_rate = drop
            middle_layer_size_multiplier = middle  # RRU
            # forget_bias = forget  # LSTM
            # feature_mask_rank = rank  # Mogrifier LSTM
            learning_rate = lr

            # We set an output path that includes the configuration, so we can later see the values in TensorBoard
            global output_path
            output_path = f"{log_path}{model_name}/{data_set_name}/{current_time}" \
                          f"/num_params{num_params}drop{drop}middle{middle}lr{lr}"  # RRU
            # f"/rounds{rounds}num_params{num_params}drop{drop}rank{rank}lr{lr}"  # Mogrifier LSTM
            # f"/num_params{num_params}drop{drop}forget{forget}lr{lr}"  # LSTM
            # f"/num_params{num_params}drop{drop}lr{lr}"  # GRU
            # f"/num_params{num_params}drop{drop}middle{middle}lr{lr}"  # RRU

            global HIDDEN_UNITS, model_function
            # Find the optimal hidden units to use without surpassing the number of parameters
            HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                     number_of_parameters=number_of_parameters,
                                                     model_function=model_function)

            # Create the model with the optimal hidden units
            model = model_function(HIDDEN_UNITS)

            # Train the model (validating after each epoch)
            model.fit(TRAINING_DATA, VALIDATION_DATA)

            # Test the last saved model (and return it's perplexity)
            return model.evaluate(TESTING_DATA)


        # To optimize multiple hyperparameters, we need to create this function that uses *args
        # https://github.com/hyperopt/hyperopt/issues/129
        def objective2(args):
            return objective(*args)


        # Create the algorithm we are going to use for hyperparameter optimization
        tpe_algo = tpe.suggest
        # Create a Trials object, so we can later print out configuration in each trial
        tpe_trials = Trials()

        # Run specified evaluations with the tpe algorithm
        tpe_best = fmin(fn=objective2, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=optimization_runs)

        print_trials_information(hyperopt_trials=tpe_trials,
                                 round_uniform=round_uniform,
                                 metric="Perplexity")
