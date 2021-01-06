import tensorflow as tf
import numpy as np

# We'll use this to measure time spent in training and testing
import time

# We'll use this to dynamically generate training event names (so each run has different name)
from datetime import datetime

# We'll use this to shuffle training data
from sklearn.utils import shuffle

# Importing some functions that will help us deal with the input data
from lm_utils import load_data, get_window_indexes, get_input_data_from_indexes, get_zeros_state

# Importing some utility functions that will help us with certain tasks
from utils import find_optimal_hidden_units, print_trainable_variables, get_batch, save_model, restore_model
from utils import print_trials_information, NetworkPrint

# This function will allow us to get information about the picked cell
from cell_registry import get_cell_information

# Importing the necessary stuff for hyperparameter optimization
from hyperopt import hp, tpe, Trials, fmin

# Importing a great optimizer
from RAdam import RAdamOptimizer

# If you have many GPUs available, you can specify which one to use here (they are indexed from 0)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# You can check if this line makes it train faster, while still training correctly
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

# Hyperparameters
# 1. Data parameters
# Choose data set to test on
data_set_name = "pennchar"  # string, one of these ["enwik8", "text8", "pennchar", "penn"]
# How many time steps we will unroll in RNN
# We do character-level 512, word-level 64
window_size = 512  # int, >= 1 (if model isn't stateful or continuous_batches) else 1 or (>= 1 and % 2 == 0)
# How many data samples we feed in a single time
batch_size = 64  # int, >= 1
# Can some of the batches be with size [batch_size, 2 * batch_size) (So we don't have as much left over data)
fixed_batch_size = False  # bool
# Should we shuffle the samples?
# We do character-level False, word-level True
shuffle_data = False  # bool
# Should batches go continuously? This might give performance boost with a stateful RNN cell
# We do character-level True, word-level False
continuous_batches = True  # bool
# 2. Model parameters
# Name of the cell you want to test
cell_name = "RRU"  # string, one of these ["RRU", "GRRUA", "GRU", "LSTM", "MogrifierLSTM"]
# Number of hidden units (This will only be used if the number_of_parameters is None or < 1)
HIDDEN_UNITS = 1024  # int, >= 1 (Probably way more than 1)
# Number of maximum allowed trainable parameters
number_of_parameters = 20000000  # int, >= 1 (Probably way more than 1)
# With what learning rate we optimize the model
learning_rate = 0.003  # float, > 0
# How many RNN layers should we have
number_of_layers = 2  # int, >= 1
# What should be the output size for the cells that have a separate output_size?
output_size = 256  # int, >= 1 (Probably way more than 1)
# Should we clip the gradients?
clip_gradients = True  # bool
# If clip_gradients = True, with what multiplier should we clip them?
clip_multiplier = 10.  # float
# What should be the embedding size for the data (how many dimensions)?
# We do PTB character-level 16, PTB word-level 64
embedding_size = 64  # int, >= 1
# Should the RNN cell be stateful
# We do character-level True, word-level False
stateful = True  # bool
# If stateful = True, you can choose a chance that zero_state will be passed instead of last state
zero_state_chance = 0.1  # float, 0 <= value <= 1
# What dropout should we apply over the RNN output
outer_dropout = 0  # float, 0 <= value <= 1
# 3. Training/testing process parameters
# How many epochs should we run?
num_epochs = 1000000  # int, >= 1
# After how many epochs with no performance gain should we early stop? (0 disabled)
break_epochs_no_gain = 7  # int, >= 0
# Should we do hyperparameter optimization?
do_hyperparameter_optimization = False   # bool
# How many runs should we run hyperparameter optimization
optimization_runs = 100  # int, >= 1
# Path, where we will save the model for further evaluating
ckpt_path = 'ckpt_lm/'  # string
# Path, where we will store ours logs
log_path = 'logdir_lm/'  # string
# After how many steps should we send the data to TensorBoard (0 - don't log after any amount of steps)
log_after_this_many_steps = 0  # integer, >= 0
# After how many steps should we print the results of training/validation/testing (0 - don't print until the last step)
print_after_this_many_steps = 1  # integer, >= 0

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


# Class for solving language modeling tasks, you can create a model, train it and test it
class LanguageModelingModel:

    def __init__(self, hidden_units):

        print("\nBuilding Graph...\n")

        tf.reset_default_graph()  # Build the graph

        # Input – batch size list of integer sequences
        x = tf.placeholder(tf.int32, shape=[None, window_size], name="x")

        # Output – batch size list of integer sequences (we'll try to predict the input sequences shifted by 1)
        y = tf.placeholder(tf.int64, shape=[None, window_size], name="y")

        # Bool value that tells us, whether or not are we in training
        training = tf.placeholder(tf.bool, name='training')

        # Outer dropout probability, so we can pass different values depending on training/testing
        outer_dropout_rate = tf.placeholder(tf.float32, name='outer_dropout_rate')

        # Instantiate our embedding matrix
        embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="word_embedding")

        # Lookup embeddings
        embed_lookup = tf.nn.embedding_lookup(embedding, x)

        # Create the RNN cell, corresponding to the one you chose above
        cells = []
        for _ in range(number_of_layers):
            if cell_name in ["RRU", "GRRUA"]:
                cell = cell_fn(hidden_units, training=training, output_size=output_size)
            else:
                cell = cell_fn(hidden_units)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        # Some cells use output size that differs from the hidden_size. And if the cell doesn't use a different
        # output_size, we need to make it as hidden units for the program to work.
        final_size = output_size
        if final_size is None:
            final_size = hidden_units

        # Extract the batch size - this allows for variable batch size
        current_batch_size = tf.shape(x)[0]

        if stateful:
            if state_is_tuple:  # LSTMs and such
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
        else:
            # Create the initial state of zeros
            initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)

        # Value will have all the outputs. State contains hidden states between the steps.
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

        # Reshape the outputs – [batch_size, window_size, final_size] -> [batch_size x window_size, final_size]
        last = tf.reshape(value, shape=(-1, final_size))

        # Apply the passed dropout
        last = tf.nn.dropout(last, rate=outer_dropout_rate)

        # Calculate the predictions. Final form – [batch_size x window_size, vocabulary_size]
        prediction = tf.matmul(last, weight) + bias
        # Reshape the predictions for easier further calculations
        # [batch_size x window_size, vocabulary_size] -> [batch_size, window_size, vocabulary_size]
        prediction = tf.reshape(prediction, shape=(-1, window_size, vocabulary_size))

        # Predictions for the second half
        half = window_size // 2
        half_prediction = prediction[:, half:, :]  # [batch_size, ceil(window_size / 2), vocabulary_size]

        # Accuracy for the second half
        half_y = y[:, half:]  # [batch_size, ceil(window_size / 2)]
        half_correct_prediction = tf.equal(tf.argmax(half_prediction, axis=-1), half_y)
        half_accuracy = tf.reduce_mean(tf.cast(half_correct_prediction, tf.float32))
        # Accuracy for the full length
        correct_prediction = tf.equal(tf.argmax(prediction, axis=-1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Calculate losses given prediction and labels
        # Loss for results (second half)
        half_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=half_prediction,
                                                                                  labels=half_y))
        # Loss for optimizing (full_length)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                                                                             labels=y))

        # Transform the loss in a format that our tasks require
        if character_level:  # Perplexity in BPC
            half_perplexity = tf.reduce_mean(half_loss)/np.log(2)
            perplexity = tf.reduce_mean(loss)/np.log(2)
        else:  # Casual perplexity
            half_perplexity = tf.exp(half_loss)
            perplexity = tf.exp(loss)

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
        tf.summary.scalar("perplexity", perplexity)

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
        self.half_perplexity = half_perplexity
        self.perplexity = perplexity
        self.half_accuracy = half_accuracy
        self.accuracy = accuracy
        # To call the optimization step (gradient descent)
        self.optimizer = optimizer

        print_trainable_variables()

        print("\nGraph Built...\n")

    def fit(self, training_data, validation_data):  # Trains the model
        tf_config = tf.ConfigProto()
        # tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        sess = tf.Session(config=tf_config)
        NetworkPrint.training_start()

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        # Adding writers so we can visualize accuracy, loss, etc. on TensorBoard
        merged_summary = tf.summary.merge_all()
        training_writer = tf.summary.FileWriter(output_path + "/training")
        training_writer.add_graph(sess.graph)

        global fixed_batch_size
        if stateful:  # If we are in stateful mode, we need fixed_batch_size, so the state shape stays the same
            fixed_batch_size = True

        if stateful or continuous_batches:
            training_indexes = get_window_indexes(len(training_data), window_size, window_size)
        else:
            training_indexes = get_window_indexes(len(training_data), window_size, window_size // 2)

        num_training_batches = len(training_indexes) // batch_size

        # Variables that help implement the early stopping if no validation perplexity decrease is observed
        epochs_no_gain = 0
        best_validation_perplexity = None

        for epoch in range(num_epochs):
            NetworkPrint.epoch_start(epoch + 1, num_epochs)

            if shuffle_data:
                training_indexes = shuffle(training_indexes)

            total_training_perplexity = 0
            total_training_accuracy = 0

            start_time = time.time()

            state = None

            extra_tasks_for_perfection = not stateful and not continuous_batches

            for i in range(num_training_batches):
                x_batch = get_batch(training_indexes, i, batch_size, fixed_batch_size, continuous_batches)

                if stateful and (np.random.uniform() < zero_state_chance or i == 0):
                    state = get_zeros_state(number_of_layers, len(x_batch), self.hidden_units, state_is_tuple)

                # Now we have batch of integers to look in text from
                x_batch, y_batch = get_input_data_from_indexes(training_data, x_batch, window_size)

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

                # Do we need to fetch the full length stats
                need_full = stateful or continuous_batches or i == 0

                if log_after_this_many_steps != 0 and i % log_after_this_many_steps == 0:
                    s, _, last_state, p, a = sess.run([merged_summary,
                                                       self.optimizer,
                                                       self.state,
                                                       self.perplexity if need_full else self.half_perplexity,
                                                       self.accuracy if need_full else self.half_accuracy],
                                                      feed_dict=feed_dict)

                    training_writer.add_summary(s, i + epoch * num_training_batches)
                else:
                    _, last_state, p, a = sess.run([self.optimizer,
                                                    self.state,
                                                    self.perplexity if need_full else self.half_perplexity,
                                                    self.accuracy if need_full else self.half_accuracy],
                                                   feed_dict=feed_dict)

                # To have 100% data covered if we had step_size = window_size // 2 , we need to have full length
                # values for first batch
                if extra_tasks_for_perfection and i == 0:
                    # Because we went through 2 halves, to have fair average we need extra addition, but we will
                    # have to divide by one more later
                    total_training_perplexity += p
                    total_training_accuracy += a

                state = last_state

                total_training_perplexity += p
                total_training_accuracy += a

                if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                        or i == num_training_batches - 1:
                    NetworkPrint.step_results(i + 1, num_training_batches, [["Perplexity", p], ["Accuracy", a]], time.time() - start_time)

            statistics_count = (num_training_batches + 1) if extra_tasks_for_perfection else num_training_batches
            average_training_perplexity = total_training_perplexity / statistics_count
            average_training_accuracy = total_training_accuracy / statistics_count
            NetworkPrint.epoch_end(epoch + 1,
                                   [["Average perplexity", average_training_perplexity],
                                    ["Average accuracy", average_training_accuracy]], time.time() - start_time)

            epoch_perplexity_summary = tf.Summary()
            epoch_perplexity_summary.value.add(tag='epoch_perplexity', simple_value=average_training_perplexity)
            training_writer.add_summary(epoch_perplexity_summary, epoch + 1)

            epoch_accuracy_summary = tf.Summary()
            epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=average_training_accuracy)
            training_writer.add_summary(epoch_accuracy_summary, epoch + 1)
            training_writer.flush()

            average_validation_perplexity = self.evaluate(validation_data, "validation", sess, epoch + 1)

            # Training and validation for epoch is done, check if validation perplexity was better this epoch
            if best_validation_perplexity is None or average_validation_perplexity < best_validation_perplexity:
                print(f"&&& New best validation perplexity - before: {best_validation_perplexity};"
                      f" after: {average_validation_perplexity} - saving model...")

                best_validation_perplexity = average_validation_perplexity

                # Save checkpoint
                save_model(sess, ckpt_path, model_name)

                epochs_no_gain = 0
            elif break_epochs_no_gain >= 1:  # Validation perplexity was worse, check break_epochs_no_gain
                epochs_no_gain += 1

                print(f"&&& No validation perplexity decrease for {epochs_no_gain} epochs,"
                      f" breaking at {break_epochs_no_gain} epochs.")

                if epochs_no_gain == break_epochs_no_gain:
                    print(f"&&& Maximum epochs without validation perplexity decrease reached, breaking...")
                    return

    def evaluate(self, data, mode="testing", session=None, iterator=1):  # Tests the model
        assert mode in ["validation", "testing"], "Mode must be \"validation\" or \"testing\""
        if mode == "validation":
            NetworkPrint.validation_start(iterator)
            sess = session
        else:
            sess = tf.Session()
            NetworkPrint.testing_start()

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Restore session
            restore_model(sess, ckpt_path)

        # Adding a writer so we can visualize accuracy, loss, etc. on TensorBoard
        writer = tf.summary.FileWriter(output_path + f"/{mode}")

        indexes = get_window_indexes(len(data), window_size, window_size // 2)

        num_batches = len(indexes) // batch_size

        total_perplexity = 0
        total_accuracy = 0

        start_time = time.time()

        for i in range(num_batches):
            x_batch = get_batch(indexes, i, batch_size, fixed_batch_size, continuous_batches)

            # Now we have batch of integers to look in text from
            x_batch, y_batch = get_input_data_from_indexes(data, x_batch, window_size)

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
                p, a = sess.run([self.perplexity, self.accuracy], feed_dict=feed_dict)
                # Because we went through 2 halves, to have fair average we need * 2, but + 2 batches also
                p = 2 * p
                a = 2 * a
            else:
                p, a = sess.run([self.half_perplexity, self.half_accuracy], feed_dict=feed_dict)

            total_perplexity += p
            total_accuracy += a

            if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                    or i == num_batches - 1:
                NetworkPrint.step_results(i + 1, num_batches, [["Average perplexity", total_perplexity / (i + 2)],
                                                               ["Average accuracy", total_accuracy / (i + 2)]],
                                          time.time() - start_time)
        average_perplexity = total_perplexity / (num_batches + 1)
        average_accuracy = total_accuracy / (num_batches + 1)
        NetworkPrint.evaluation_end(mode, [["Average perplexity", average_perplexity],
                                           ["Average accuracy", average_accuracy]],
                                    time.time() - start_time)

        # We add this to TensorBoard so we don't have to dig in console logs and nohups
        perplexity_summary = tf.Summary()
        perplexity_summary.value.add(tag=f'{mode}_perplexity', simple_value=average_perplexity)
        writer.add_summary(perplexity_summary, 1)

        accuracy_summary = tf.Summary()
        accuracy_summary.value.add(tag=f'{mode}_accuracy', simple_value=average_accuracy)
        writer.add_summary(accuracy_summary, 1)
        writer.flush()

        return average_perplexity


if __name__ == '__main__':  # Main function

    TRAINING_DATA, VALIDATION_DATA, TESTING_DATA, vocabulary_size = load_data(data_set_name)  # Load data set

    # To see how it trains in small amounts (If it's usually in a 90%/5%/5% split, now it's in a 1%/1%/1% split)
    '''
    TRAINING_DATA = TRAINING_DATA[:len(TRAINING_DATA) // 90]
    VALIDATION_DATA = VALIDATION_DATA[:len(VALIDATION_DATA) // 5]
    TESTING_DATA = TESTING_DATA[:len(TESTING_DATA) // 5]
    '''

    # From which function/class we can get the model
    model_function = LanguageModelingModel

    if not do_hyperparameter_optimization:
        HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                 number_of_parameters=number_of_parameters,
                                                 model_function=model_function)

        MODEL = model_function(HIDDEN_UNITS)  # Create the model

        MODEL.fit(TRAINING_DATA, VALIDATION_DATA)  # Train the model (validating after each epoch)

        MODEL.evaluate(TESTING_DATA)  # Test the last saved model
    else:
        # hp.choice
        batch_choice = [32, 64, 128]
        num_layers_choice = [1, 2, 3]
        # We need this, so we can print the hp.choice answers normally
        choices = {
            'batch': batch_choice,
            'num_layers': num_layers_choice
        }

        # What to do with hp.uniforms that need to be rounded
        round_uniform = ['num_params', 'out_size']

        space = [
            # hp.choice
            hp.choice('batch', batch_choice),
            hp.choice('num_layers', num_layers_choice),
            # hp.uniform
            hp.uniform('num_params', 12000000, 28000000),
            hp.uniform('out_size', 128, 256),
            # hp.loguniform
            hp.loguniform('lr', np.log(0.0001), np.log(0.009)),
        ]


        def objective(batch, num_layers, num_params, out_size, lr):
            # For some values we need extra stuff
            num_params = round(num_params)
            out_size = round(out_size)

            # We'll optimize these parameters
            global batch_size, number_of_layers, number_of_parameters, learning_rate, output_size
            batch_size = batch
            number_of_layers = num_layers
            number_of_parameters = num_params
            output_size = out_size
            learning_rate = lr

            global output_path
            output_path = f"{log_path}{model_name}/{data_set_name}/{current_time}" \
                          f"/batch{batch}num_layers{num_layers}num_params{num_params}out_size{out_size}lr{lr}"

            global HIDDEN_UNITS, model_function

            HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                     number_of_parameters=number_of_parameters,
                                                     model_function=model_function)

            model = model_function(HIDDEN_UNITS)  # Create the model

            model.fit(TRAINING_DATA, VALIDATION_DATA)  # Train the model (validating after each epoch)

            return model.evaluate(TESTING_DATA)  # Test the last saved model (it returns testing perplexity)

        # https://github.com/hyperopt/hyperopt/issues/129
        def objective2(args):
            return objective(*args)

        # Create the algorithm
        tpe_algo = tpe.suggest
        # Create trials object
        tpe_trials = Trials()

        # Run specified evaluations with the tpe algorithm
        tpe_best = fmin(fn=objective2, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=optimization_runs)

        print_trials_information(tpe_trials,
                                 hyperopt_choices=choices,
                                 round_uniform=round_uniform,
                                 metric="Perplexity")
