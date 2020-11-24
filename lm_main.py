import tensorflow as tf
import numpy as np
import time
from datetime import datetime  # We'll use this to dynamically generate training event names

from sklearn.utils import shuffle  # We'll use this to shuffle training data

# Importing some functions that will help us deal with the input data
from lm_utils import load_data
from lm_utils import get_window_indexes
from lm_utils import get_input_data_from_indexes

# Importing some utility functions that will help us with certain tasks
from utils import find_optimal_hidden_units
from utils import gelu
from utils import print_trainable_variables
# Importing the necessary stuff for hyperparameter optimization
from hyperopt import hp, tpe, Trials, fmin

# Importing fancier optimizer(s)
# from RAdam import RAdamOptimizer
from adam_decay import AdamOptimizer_decay

# If you have many GPUs available, you can specify which one to use here (they are indexed from 0)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# You can check if this line makes it train faster, while still training correctly
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

# Choose your cell
cell_name = "RRU3"  # Here you can type in the name of the cell you want to use

# Maybe we can put these in a separate file called cells.py or something, and import it
output_size = None  # Most cells don't have an output size, so we by default set it as None
state_is_tuple = False  # So we can make the RNN cell stateful even for LSTM based cells such as LSTM and MogrifierLSTM
has_training_bool = False
if cell_name == "RRU1":  # ReZero version
    from cells.RRUCell import RRUCell
    cell_fn = RRUCell
    output_size = 256
    has_training_bool = True
    model_name = 'rru_model'

elif cell_name == "RRU2":  # Gated version with 1 transformation
    from cells.GatedRRUCell import RRUCell  # (I already have some results with this one)
    cell_fn = RRUCell
    has_training_bool = True
    model_name = 'grru_model'

elif cell_name == "RRU3":  # Gated version with separate output size
    from cells.GatedRRUCell_a import RRUCell
    cell_fn = RRUCell
    has_training_bool = True
    output_size = 256
    model_name = "grrua_model"  # We have hopes for this one

elif cell_name == "GRU":
    from cells.GRUCell import GRUCell
    cell_fn = GRUCell
    model_name = 'gru_model'

elif cell_name == "LSTM":
    from cells.BasicLSTMCell import BasicLSTMCell
    cell_fn = BasicLSTMCell
    state_is_tuple = True
    model_name = 'lstm_model'

elif cell_name == "MogrifierLSTM":  # For this you have to have dm-sonnet, etc. installed
    from cells.MogrifierLSTMCell import MogrifierLSTMCell
    cell_fn = MogrifierLSTMCell
    state_is_tuple = True
    model_name = 'mogrifier_lstm_model'

else:
    raise ValueError(f"No such cell ('{cell_name}') has been implemented!")

# Hyperparameters
# Data parameters
data_set_name = "pennchar"  # "enwik8" | "text8" | "pennchar" | "penn" (which data set to test on)
character_level = True
if data_set_name in ["penn"]:
    character_level = False
vocabulary_size = None  # We will load this from a pickle file, so changing this here won't do a thing
window_size = 512  # This must be 1 or an even number (otherwise loss calculation won't be correct (we do half_loss))
assert window_size == 1 or window_size % 2 == 0, "Variable window_size must be 1 or an even number!"
batch_size = 64
fixed_batch_size = False  # With this False it may run some batches on size [batch_size, 2 * batch_size)
continuous_batches = True  # Batches go continuously, this might give performance boost with a stateful RNN cell
shuffle_data = False  # Should we shuffle the samples?
# Training
num_epochs = 1000000  # We can code this to go infinity, but I see no point, we won't wait 1 million epochs anyway
# If validation perplexity doesn't get lower, after how many epochs we should break (-1 -> disabled)
break_epochs_no_gain = 3
number_of_parameters = 24000000  # 24 million learnable parameters
HIDDEN_UNITS = 1024  # This will only be used if the number_of_parameters is None or < 1
embedding_size = 128
learning_rate = 0.001  # At 0,001 LSTM and GRU explodes a bit, and at 0.0001 Mogrifier LSTM can't learn, so 0,0005!
number_of_layers = 2
stateful = True  # Should the RNN cell be stateful? If True, you can modify it's zero_state_chance below.
zero_state_chance = 0.1  # Chance that zero_state is passed instead of last state (I don't know what value is best yet)
outer_dropout = 0  # 0, if you do not want outer dropout
do_hyperparameter_optimization = False

ckpt_path = 'ckpt_lm/'
log_path = 'logdir_lm/'

# After how many steps should we send the data to TensorBoard (0 - don't log after any amount of steps)
log_after_this_many_steps = 0
assert log_after_this_many_steps >= 0, "Invalid value for variable log_after_this_many_steps, it must be >= 0!"
# After how many steps should we print the results of training/validation/testing (0 - don't print until the last step)
print_after_this_many_steps = 100
assert print_after_this_many_steps >= 0, "Invalid value for variable print_after_this_many_steps, it must be >= 0!"

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = log_path + model_name + f'/{data_set_name}/' + current_time


class LMModel:

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
            if has_training_bool:
                cell = cell_fn(hidden_units, training=training)
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

        # Apply GELU activation
        value = gelu(value)

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
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        optimizer = AdamOptimizer_decay(learning_rate=learning_rate,
                                        L2_decay=0.01,
                                        decay_vars=decay_vars).minimize(loss)
        # optimizer = RAdamOptimizer(learning_rate=learning_rate, L2_decay=0.0, epsilon=1e-8).minimize(loss)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

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

    def fit(self, training_data, validation_data=None):
        tf_config = tf.ConfigProto()
        # tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with tf.Session(config=tf_config) as sess:
            print("|*|*|*|*|*| Starting training... |*|*|*|*|*|")

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Adding writers so we can visualize accuracy, loss, etc. on TensorBoard
            merged_summary = tf.summary.merge_all()
            training_writer = tf.summary.FileWriter(output_path + "/training")
            training_writer.add_graph(sess.graph)

            validation_writer = tf.summary.FileWriter(output_path + "/validation")
            validation_writer.add_graph(sess.graph)

            # When stateful (maybe in other cases) we want the data to be continuous, so we need them to be exactly one
            # after the other

            if stateful:  # If we are in stateful mode, we need fixed_batch_size, so the state shape stays the same
                global fixed_batch_size
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
                print(f"------ Epoch {epoch + 1} out of {num_epochs} ------")

                if shuffle_data:
                    training_indexes = shuffle(training_indexes)

                total_training_perplexity = 0
                total_training_accuracy = 0

                start_time = time.time()

                state = None

                for i in range(num_training_batches):
                    if continuous_batches:
                        x_batch = []
                        if fixed_batch_size:
                            for j in range(i, num_training_batches * batch_size, num_training_batches):
                                x_batch.append(training_indexes[j])
                        else:
                            for j in range(i, len(training_indexes), num_training_batches):
                                x_batch.append(training_indexes[j])
                    else:
                        # The batch_size is fixed or it's not the last
                        if fixed_batch_size or i != num_training_batches - 1:
                            x_batch = training_indexes[i * batch_size: i * batch_size + batch_size]
                        else:
                            # Run the remaining sequences (that might not be exactly batch_size (they might be larger))
                            x_batch = training_indexes[i * batch_size:]

                    if stateful and (np.random.uniform() < zero_state_chance or i == 0):
                        # state = np.zeros((number_of_layers, len(x_batch), hidden_units))
                        zero_state = np.zeros((len(x_batch), self.hidden_units))
                        if state_is_tuple:
                            zero_state = [zero_state, zero_state]
                        state = []
                        for j in range(number_of_layers):
                            state.append(zero_state)

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
                    extra_tasks_for_perfection = not stateful and not continuous_batches and i == 0
                    if extra_tasks_for_perfection:
                        # Because we went through 2 halves, to have fair average we need extra addition, but we will
                        # have to divide by one more later
                        total_training_perplexity += p
                        total_training_accuracy += a

                    state = last_state

                    total_training_perplexity += p
                    total_training_accuracy += a

                    if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                            or i == num_training_batches - 1:
                        print(f"Step {i + 1} of {num_training_batches} | "
                              f"Perplexity: {p}, "
                              f"Accuracy: {a}, "
                              f"Time from start: {time.time() - start_time}")

                statistics_count = (num_training_batches + 1) if extra_tasks_for_perfection else num_training_batches
                average_training_perplexity = total_training_perplexity / statistics_count
                average_training_accuracy = total_training_accuracy / statistics_count
                print(f"   Epoch {epoch + 1} | "
                      f"Average perplexity: {average_training_perplexity}, "
                      f"Average accuracy: {average_training_accuracy}, "
                      f"Time spent: {time.time() - start_time}")

                epoch_perplexity_summary = tf.Summary()
                epoch_perplexity_summary.value.add(tag='epoch_perplexity', simple_value=average_training_perplexity)
                training_writer.add_summary(epoch_perplexity_summary, epoch + 1)

                epoch_accuracy_summary = tf.Summary()
                epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=average_training_accuracy)
                training_writer.add_summary(epoch_accuracy_summary, epoch + 1)
                training_writer.flush()

                if validation_data is not None:
                    print(f"------ Starting validation for epoch {epoch + 1} out of {num_epochs}... ------")

                    validation_indexes = get_window_indexes(len(validation_data), window_size, window_size // 2)

                    num_validation_batches = len(validation_indexes) // batch_size

                    total_validation_perplexity = 0
                    total_validation_accuracy = 0

                    start_time = time.time()

                    for i in range(num_validation_batches):
                        if continuous_batches:
                            x_batch = []
                            if fixed_batch_size:
                                for j in range(i, num_validation_batches * batch_size, num_validation_batches):
                                    x_batch.append(validation_indexes[j])
                            else:
                                for j in range(i, len(validation_indexes), num_validation_batches):
                                    x_batch.append(validation_indexes[j])
                        else:
                            # If the batch size is fixed or it's not the last batch, we use batch size
                            if fixed_batch_size or i != num_validation_batches - 1:
                                x_batch = validation_indexes[i * batch_size: i * batch_size + batch_size]
                            else:
                                # Run the remaining sequences (that might be larger than batch size)
                                x_batch = validation_indexes[i * batch_size:]

                        if stateful:
                            # state = np.zeros((number_of_layers, len(x_batch), hidden_units))
                            zero_state = np.zeros((len(x_batch), self.hidden_units))
                            if state_is_tuple:
                                zero_state = [zero_state, zero_state]
                            state = []
                            for j in range(number_of_layers):
                                state.append(zero_state)

                        # Now we have batch of integers to look in text from
                        x_batch, y_batch = get_input_data_from_indexes(validation_data, x_batch, window_size)

                        if stateful:
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

                        total_validation_perplexity += p
                        total_validation_accuracy += a

                        if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                                or i == num_validation_batches - 1:
                            print(f"Step {i + 1} of {num_validation_batches} | "
                                  f"Average perplexity: {total_validation_perplexity / (i + 2)}, "
                                  f"Average accuracy: {total_validation_accuracy / (i + 2)}, "
                                  f"Time from start: {time.time() - start_time}")

                    # + 1 because we counted the 1st batch twice
                    average_validation_perplexity = total_validation_perplexity / (num_validation_batches + 1)
                    average_validation_accuracy = total_validation_accuracy / (num_validation_batches + 1)
                    print(f"Final validation stats | "
                          f"Average perplexity: {average_validation_perplexity}, "
                          f"Average accuracy: {average_validation_accuracy}, "
                          f"Time spent: {time.time() - start_time}")

                    epoch_perplexity_summary = tf.Summary()
                    epoch_perplexity_summary.value.add(tag='epoch_perplexity', simple_value=average_validation_perplexity)
                    validation_writer.add_summary(epoch_perplexity_summary, epoch + 1)

                    epoch_accuracy_summary = tf.Summary()
                    epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=average_validation_accuracy)
                    validation_writer.add_summary(epoch_accuracy_summary, epoch + 1)
                    validation_writer.flush()

                    # Training and validation for epoch is done, check if validation perplexity was better this epoch
                    if best_validation_perplexity is None or average_validation_perplexity < best_validation_perplexity:
                        print(f"&&& New best validation perplexity - before: {best_validation_perplexity};"
                              f" after: {average_validation_perplexity} - saving model...")

                        best_validation_perplexity = average_validation_perplexity

                        # Save checkpoint
                        saver = tf.compat.v1.train.Saver()
                        saver.save(sess, ckpt_path + model_name + ".ckpt")

                        epochs_no_gain = 0
                    elif break_epochs_no_gain >= 1:  # Validation perplexity was worse, check break_epochs_no_gain
                        epochs_no_gain += 1

                        print(f"&&& No validation perplexity decrease for {epochs_no_gain} epochs,"
                              f" breaking at {break_epochs_no_gain} epochs.")

                        if epochs_no_gain == break_epochs_no_gain:
                            print(f"&&& Maximum epochs without validation perplexity decrease reached, breaking...")
                            break  # Probably return would do the same thing (for now)
            # Training ends here
            '''We used to save here - model saved after the last epoch'''
            # Save checkpoint
            # saver = tf.compat.v1.train.Saver()
            # saver.save(sess, ckpt_path + model_name + ".ckpt")  # global_step=1 and etc., can index the saved model

    def evaluate(self, data):
        with tf.Session() as sess:
            print("|*|*|*|*|*| Starting testing... |*|*|*|*|*|")

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Adding a writer so we can visualize accuracy, loss, etc. on TensorBoard
            testing_writer = tf.summary.FileWriter(output_path + "/testing")
            testing_writer.add_graph(sess.graph)

            indexes = get_window_indexes(len(data), window_size, window_size // 2)

            # Restore session
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver = tf.compat.v1.train.Saver()
            # If there is a correct checkpoint at the path restore it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            num_batches = len(indexes) // batch_size

            total_perplexity = 0
            total_accuracy = 0

            start_time = time.time()

            for i in range(num_batches):
                if continuous_batches:
                    x_batch = []
                    if fixed_batch_size:
                        for j in range(i, num_batches * batch_size, num_batches):
                            x_batch.append(indexes[j])
                    else:
                        for j in range(i, len(indexes), num_batches):
                            x_batch.append(indexes[j])
                else:
                    if fixed_batch_size or i != num_batches - 1:  # The batch_size is fixed or it's not the last
                        x_batch = indexes[i * batch_size: i * batch_size + batch_size]
                    else:
                        # Run the remaining sequences (that might not be exactly batch_size (they might be larger))
                        x_batch = indexes[i * batch_size:]

                if stateful:
                    # state = np.zeros((number_of_layers, len(x_batch), hidden_units))
                    zero_state = np.zeros((len(x_batch), self.hidden_units))
                    if state_is_tuple:
                        zero_state = [zero_state, zero_state]
                    state = []
                    for j in range(number_of_layers):
                        state.append(zero_state)

                # Now we have batch of integers to look in text from
                x_batch, y_batch = get_input_data_from_indexes(data, x_batch, window_size)

                if stateful:
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
                    print(f"Step {i + 1} of {num_batches} | "
                          f"Average perplexity: {total_perplexity / (i + 2)}, "
                          f"Average accuracy: {total_accuracy / (i + 2)}, "
                          f"Time from start: {time.time() - start_time}")
            average_perplexity = total_perplexity / (num_batches + 1)
            average_accuracy = total_accuracy / (num_batches + 1)
            print(f"Final testing stats | "
                  f"Average perplexity: {average_perplexity}, "
                  f"Average accuracy: {average_accuracy}, "
                  f"Time spent: {time.time() - start_time}")

            # We add this to TensorBoard so we don't have to dig in console logs and nohups
            testing_perplexity_summary = tf.Summary()
            testing_perplexity_summary.value.add(tag='testing_perplexity', simple_value=average_perplexity)
            testing_writer.add_summary(testing_perplexity_summary, 1)

            testing_accuracy_summary = tf.Summary()
            testing_accuracy_summary.value.add(tag='testing_accuracy', simple_value=average_accuracy)
            testing_writer.add_summary(testing_accuracy_summary, 1)
            testing_writer.flush()

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
    model_function = LMModel

    if not do_hyperparameter_optimization:
        HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                 number_of_parameters=number_of_parameters,
                                                 model_function=model_function)

        MODEL = model_function(HIDDEN_UNITS)  # Create the model

        MODEL.fit(TRAINING_DATA, VALIDATION_DATA)  # Train the model (validating after each epoch)

        MODEL.evaluate(TESTING_DATA)  # Test the last saved model
    else:
        times_to_evaluate = 2

        lr_choice = [0.1, 0.05, 0.01, 0.005, 0.001]
        num_layers_choice = [1, 2, 3]
        batch_choice = [1, 2, 4, 8, 16, 32, 64]
        # We need this, so we can print the hp.choice answers normally
        choices = {
            'lr': lr_choice,
            'num_layers': num_layers_choice,
            'batch': batch_choice
        }

        space = [
            hp.choice('lr', lr_choice),
            hp.choice('num_layers', num_layers_choice),
            hp.choice('batch', batch_choice)
        ]


        def objective(lr, num_layers, batch):
            # The parameters to be optimized
            global learning_rate
            learning_rate = lr
            global number_of_layers
            number_of_layers = num_layers
            global batch_size
            batch_size = batch

            # This might give some clues
            global output_path
            output_path = f"{log_path}{model_name}/{data_set_name}/{current_time}/lr{lr}layers{num_layers}batch{batch}"

            global HIDDEN_UNITS
            global model_function

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

        # Run 2000 evals with the tpe algorithm
        tpe_best = fmin(fn=objective2, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=times_to_evaluate)

        from utils import print_trials_information

        print_trials_information(tpe_trials, choices, metric="Perplexity")
