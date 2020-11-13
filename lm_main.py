# If you have many GPUs available, you can specify which one to use here (they are indexed from 0)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# You can check if this line makes it train faster, while still training correctly
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

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

# Importing fancier optimizer(s)
# from RAdam import RAdamOptimizer
from adam_decay import AdamOptimizer_decay

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
vocabulary_size = None  # We will load this from a pickle file, so changing this here won't do a thing
window_size = 512
step_size = window_size // 2
batch_size = 64  # Enwik8 - 64. PTB character-level 128?
fixed_batch_size = False  # With this False it may run some batches on size [batch_size, 2 * batch_size)
continuous_batches = True  # Batches go continuously, this might give performance boost with a stateful RNN cell
shuffle_data = False  # Should we shuffle the samples?
# Training
num_epochs = 1000000  # We can code this to go infinity, but I see no point, we won't wait 1 million epochs anyway
break_epochs_no_gain = 3  # If validation BPC doesn't get lower, after how many epochs we should break (-1 -> disabled)
number_of_parameters = 24000000  # 24 million learnable parameters
HIDDEN_UNITS = 1024  # This will only be used if the number_of_parameters is None or < 1
embedding_size = 128
learning_rate = 0.001  # At 0,001 LSTM and GRU explodes a bit, and at 0.0001 Mogrifier LSTM can't learn, so 0,0005!
number_of_layers = 2
stateful = True  # Should the RNN cell be stateful? If True, you can modify it's zero_state_chance below.
zero_state_chance = 0.1  # Chance that zero_state is passed instead of last state (I don't know what value is best yet)
outer_dropout = 0  # 0 if you do not want outer dropout

ckpt_path = 'ckpt_lm/'
log_path = 'logdir_lm/'

# After how many steps should we send the data to TensorBoard (0 - don't log after any amount of steps)
log_after_this_many_steps = 0
assert log_after_this_many_steps >= 0, "Invalid value for variable log_after_this_many_steps, it must be >= 0!"
# After how many steps should we print the results of training/validating/testing (0 - don't print until the last step)
print_after_this_many_steps = 100
assert print_after_this_many_steps >= 0, "Invalid value for variable print_after_this_many_steps, it must be >= 0!"

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = log_path + model_name + f'/{data_set_name}/' + current_time


class LMModel:

    def __init__(self, hidden_units):

        print("\nBuilding Graph...\n")

        tf.reset_default_graph()  # Build the graph

        # Batch size list of integer sequences
        x = tf.placeholder(tf.int32, shape=[None, window_size], name="x")

        # Labels for word prediction
        y = tf.placeholder(tf.int64, shape=[None, window_size], name="y")

        # Bool value if we are in training or not
        training = tf.placeholder(tf.bool, name='training')

        # Output drop probability so we can pass different values depending on training/ testing
        outer_dropout_rate = tf.placeholder(tf.float32, name='outer_dropout_rate')

        # Instantiate our embedding matrix
        embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="word_embedding")

        # Lookup embeddings
        embed_lookup = tf.nn.embedding_lookup(embedding, x)

        # Create the RNN cell, corresponding to the one you chose, for example, RRU, GRU, LSTM, MogrifierLSTM
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
            # Magic here
            if state_is_tuple:  # LSTMs and such
                initial_state = tf.placeholder(tf.float32, shape=[number_of_layers, 2, None, hidden_units], name="initial_state")
                initial_state = tf.unstack(initial_state, axis=0)
                initial_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(initial_state[idx][0], initial_state[idx][1])
                     for idx in range(number_of_layers)]
                )
            else:  # Regular shaped RRN cells
                initial_state = tf.placeholder(tf.float32, shape=[number_of_layers, None, hidden_units], name="initial_state")
                initial_state = tuple(tf.unstack(initial_state, axis=0))
        else:
            # Create the initial state of zeros
            initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)
            #
            # I don't think we need these bottom two lines anymore
            # initial_state += np.asarray([1.0, -1.0] * (initial_state.get_shape().as_list()[-1] // 2)) * 0.1
            # initial_state += 0.1

        # Wrap our cell in a dropout wrapper
        # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.85)

        # Value will have all the outputs. State contains hidden states between the steps.
        # embed_lookup = tf.unstack(embed_lookup, axis=1)
        value, state = tf.nn.dynamic_rnn(cell,
                                         embed_lookup,
                                         initial_state=initial_state,
                                         dtype=tf.float32)

        # ### Maybe we can do this later if we use tf2.0 final_size = tf.shape(value)[-1]

        ''' Uncomment these if you want a lot of extra data (it takes a lot of memory space, though) '''
        # value = value_all[:, :, :final_size]
        # gate = value_all[:, :, final_size:final_size+hidden_units]
        # hidden_mem = value_all[:, :, final_size + hidden_units:]
        # # Instantiate weights
        # gate_img = tf.expand_dims(gate[0:1, :, :], -1)
        # tf.summary.image("gate", tf.transpose(gate_img, [0, 2, 1, 3]), max_outputs=16)
        # tf.summary.histogram("gate", gate_img)
        # hidden_img = tf.expand_dims(hidden_mem[0:1, :, :], -1)
        # tf.summary.image("state", tf.transpose(hidden_img, [0, 2, 1, 3]), max_outputs=16)
        # tf.summary.histogram("state", hidden_img)
        # tf.summary.scalar("candWeight", cell.candidate_weight)
        # tf.summary.histogram("s_mul", tf.sigmoid(cell.S_bias_variable)*1.5)
        # tf.summary.scalar("stateWeight", cell.prev_state_weight)
        # tf.summary.scalar("W_mul", cell._W_mul)

        weight = tf.get_variable("output", [final_size, vocabulary_size])
        # Instantiate biases
        bias = tf.Variable(tf.constant(0.0, shape=[vocabulary_size]))

        # [batch_size, window_size, final_size] -> [batch_size x window_size, final_size]
        # value = tf.stack(value, axis=1)
        # value -= tf.reduce_mean(value, [-1], keepdims=True)
        # value = instance_norm(value)
        value = gelu(value)

        last = tf.reshape(value, shape=(-1, final_size))

        last = tf.nn.dropout(last, rate=outer_dropout_rate)

        # [batch_size, window_size] -> [batch_size x window_size]
        labels = tf.reshape(y, [-1])

        # Final form should be [batch_size x window_size, vocabulary_size]
        prediction = tf.matmul(last, weight) + bias  # What we actually do is calculate the loss over the batch

        ''' Last half accuracy predictions '''
        half = window_size // 2
        half_last = tf.reshape(value[:, half:, :], shape=(-1, final_size))
        half_prediction = tf.matmul(half_last, weight) + bias
        half_y = y[:, half:]
        half_correct_prediction = tf.equal(tf.argmax(half_prediction, axis=1), tf.reshape(half_y, [-1]))
        half_accuracy = tf.reduce_mean(tf.cast(half_correct_prediction, tf.float32))
        tf.summary.scalar("half_accuracy", half_accuracy)
        ''' Full size accuracy predictions '''
        correct_prediction = tf.equal(tf.argmax(prediction, axis=1), labels)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

        # Calculate the loss given prediction and labels
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=labels))

        tf.summary.scalar("loss", loss)

        bpc = tf.reduce_mean(loss)/np.log(2)

        tf.summary.scalar("bpc", bpc)

        perplexity = tf.exp(loss)

        tf.summary.scalar("perplexity", perplexity)

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
        self.loss = loss
        self.perplexity = perplexity
        self.bpc = bpc
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.prediction = prediction
        self.correct_prediction = correct_prediction

        print_trainable_variables()

        print("\nGraph Built...\n")

    def fit(self, train_data, valid_data=None):
        tf_config = tf.ConfigProto()
        # tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with tf.Session(config=tf_config) as sess:
            print("|*|*|*|*|*| Starting training... |*|*|*|*|*|")

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Adding a writer so we can visualize accuracy and loss on TensorBoard
            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(output_path)
            train_writer.add_graph(sess.graph)

            validation_writer = tf.summary.FileWriter(output_path + "/validation")
            validation_writer.add_graph(sess.graph)

            # When stateful and in other times, we want the data to be continuous, so we need them to be exactly
            # one after the other

            if stateful:  # Need to think about this later, but we need this so we can take the last state, so it
                # doesn't have a different batch size. We can think about other ways to deal with this later
                global fixed_batch_size
                fixed_batch_size = True

            if stateful or continuous_batches:
                indexes = get_window_indexes(len(train_data), window_size, window_size)
            else:
                indexes = get_window_indexes(len(train_data), window_size, step_size)

            num_batches = len(indexes) // batch_size

            # Variables that help implement the early stopping if no validation perplexity (in BPC) decrease is observed
            epochs_no_gain = 0
            best_validation_bpc = None

            for epoch in range(num_epochs):
                print(f"------ Epoch {epoch + 1} out of {num_epochs} ------")

                if shuffle_data:
                    indexes = shuffle(indexes)

                total_loss = 0
                total_accuracy = 0
                total_bpc = 0
                total_perplexity = 0

                start_time = time.time()

                # At first I wanted to get the zero state from this, but it won't work, because it's a tensor. I can
                # make the user input the zero_state himself, though. We might be able to allow to input their own zero
                # state version at the top of the page
                # zero_state = cell_fn(hidden_units).zero_state(batch_size, dtype=tf.float32)
                ''' Might have some small speed increase by doing something like this if you want perfect code :D
                zero_state_fixed_single = np.zeros((batch_size, hidden_units))
                zero_state_fixed = []
                for i in range(number_of_layers):
                    zero_state_fixed.append(zero_state_fixed_single)
                '''

                state = None

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

                    if stateful and (np.random.uniform() < zero_state_chance or i == 0):
                        # state = np.zeros((number_of_layers, len(x_batch), hidden_units))
                        zero_state = np.zeros((len(x_batch), self.hidden_units))
                        if state_is_tuple:
                            zero_state = [zero_state, zero_state]
                        state = []
                        for j in range(number_of_layers):
                            state.append(zero_state)

                    # Now we have batch of integers to look in text from
                    x_batch, y_batch = get_input_data_from_indexes(train_data, x_batch, window_size)

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

                    if log_after_this_many_steps != 0 and i % log_after_this_many_steps == 0:
                        s, _, last_state, l, p, b, a = sess.run([merged_summary,
                                                                 self.optimizer,
                                                                 self.state,
                                                                 self.loss,
                                                                 self.perplexity,
                                                                 self.bpc,
                                                                 self.accuracy],
                                                                feed_dict=feed_dict)

                        train_writer.add_summary(s, i + epoch * num_batches)
                    else:
                        _, last_state, l, p, b, a = sess.run([self.optimizer,
                                                              self.state,
                                                              self.loss,
                                                              self.perplexity,
                                                              self.bpc,
                                                              self.accuracy],
                                                             feed_dict=feed_dict)

                    state = last_state

                    total_loss += l
                    if i == 0:
                        total_accuracy = a
                        total_bpc = b
                        total_perplexity = p
                    else:
                        total_accuracy = (total_accuracy * i + a) / (i + 1)
                        total_bpc = (total_bpc * i + b) / (i + 1)
                        total_perplexity = (total_perplexity * i + p) / (i + 1)

                    if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                            or i == num_batches - 1:
                        print(f"Step {i + 1} of {num_batches} | Loss: {l}, Perplexity: {p}, BPC: {b}, Accuracy: {a},"
                              f" TimeFromStart: {time.time() - start_time}")

                print(f"   Epoch {epoch + 1} | Loss: {total_loss}, Perplexity: {total_perplexity}, BPC: {total_bpc},"
                      f" Accuracy: {total_accuracy}, TimeSpent: {time.time() - start_time}")

                epoch_accuracy_summary = tf.Summary()
                epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=total_accuracy)
                train_writer.add_summary(epoch_accuracy_summary, epoch + 1)

                epoch_bpc_summary = tf.Summary()
                epoch_bpc_summary.value.add(tag='epoch_bpc', simple_value=total_bpc)
                train_writer.add_summary(epoch_bpc_summary, epoch + 1)

                if valid_data is not None:
                    print(f"------ Starting validation for epoch {epoch + 1} out of {num_epochs}... ------")

                    if stateful or continuous_batches:
                        validation_indexes = get_window_indexes(len(valid_data), window_size, window_size)
                    else:
                        validation_indexes = get_window_indexes(len(valid_data), window_size, step_size)

                    num_validation_batches = len(validation_indexes) // batch_size

                    total_val_loss = 0
                    total_val_accuracy = 0
                    total_val_bpc = 0
                    total_val_perplexity = 0

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

                        if stateful and (np.random.uniform() < zero_state_chance or i == 0):
                            # state = np.zeros((number_of_layers, len(x_batch), hidden_units))
                            zero_state = np.zeros((len(x_batch), self.hidden_units))
                            if state_is_tuple:
                                zero_state = [zero_state, zero_state]
                            state = []
                            for j in range(number_of_layers):
                                state.append(zero_state)

                        # Now we have batch of integers to look in text from
                        x_batch, y_batch = get_input_data_from_indexes(valid_data, x_batch, window_size)

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

                        last_state, l, p, b, a = sess.run([self.state,
                                                           self.loss,
                                                           self.perplexity,
                                                           self.bpc,
                                                           self.accuracy],
                                                          feed_dict=feed_dict)

                        state = last_state

                        total_val_loss += l
                        if i == 0:
                            total_val_accuracy = a
                            total_val_bpc = b
                            total_val_perplexity = p
                        else:
                            total_val_accuracy = (total_val_accuracy * i + a) / (i + 1)
                            total_val_bpc = (total_val_bpc * i + b) / (i + 1)
                            total_val_perplexity = (total_val_perplexity * i + p) / (i + 1)

                        if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                                or i == num_validation_batches - 1:
                            print(f"Step {i + 1} of {num_validation_batches} | Loss: {total_val_loss},"
                                  f" Perplexity: {total_val_perplexity}, BPC: {total_val_bpc},"
                                  f" Accuracy: {total_val_accuracy}, TimeFromStart: {time.time() - start_time}")

                    print(f"Final validation stats | Loss: {total_val_loss}, Perplexity: {total_val_perplexity},"
                          f" BPC: {total_val_bpc}, Accuracy: {total_val_accuracy},"
                          f" TimeSpent: {time.time() - start_time}")

                    epoch_accuracy_summary = tf.Summary()
                    epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=total_val_accuracy)
                    validation_writer.add_summary(epoch_accuracy_summary, epoch + 1)

                    epoch_bpc_summary = tf.Summary()
                    epoch_bpc_summary.value.add(tag='epoch_bpc', simple_value=total_val_bpc)
                    validation_writer.add_summary(epoch_bpc_summary, epoch + 1)
                    validation_writer.flush()

                    '''Here the training and validation epoch have been run, now get the lowest validation bpc model'''
                    # Check if validation perplexity was better
                    if best_validation_bpc is None or total_val_bpc < best_validation_bpc:
                        print(f"&&& New best validation bpc - before: {best_validation_bpc};"
                              f" after: {total_val_bpc} - saving model...")

                        best_validation_bpc = total_val_bpc

                        # Save checkpoint
                        saver = tf.compat.v1.train.Saver()
                        saver.save(sess, ckpt_path + model_name + ".ckpt")

                        epochs_no_gain = 0
                    elif break_epochs_no_gain >= 1:  # Validation BPC was worse. Check if break_epochs_no_gain is on
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

            if stateful or continuous_batches:
                indexes = get_window_indexes(len(data), window_size, window_size)
            else:
                indexes = get_window_indexes(len(data), window_size, step_size)

            # Restore session
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver = tf.compat.v1.train.Saver()
            # If there is a correct checkpoint at the path restore it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            num_batches = len(indexes) // batch_size

            total_loss = 0
            total_accuracy = 0
            total_bpc = 0
            total_perplexity = 0

            start_time = time.time()

            state = None

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

                if stateful and (np.random.uniform() < zero_state_chance or i == 0):
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

                last_state, l, p, b, a = sess.run([self.state, self.loss, self.perplexity, self.bpc, self.accuracy],
                                                  feed_dict=feed_dict)

                state = last_state

                total_loss += l
                if i == 0:
                    total_accuracy = a
                    total_bpc = b
                    total_perplexity = p
                else:
                    total_accuracy = (total_accuracy * i + a) / (i + 1)
                    total_bpc = (total_bpc * i + b) / (i + 1)
                    total_perplexity = (total_perplexity * i + p) / (i + 1)

                if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                        or i == num_batches - 1:
                    print(f"Step {i + 1} of {num_batches} | Loss: {total_loss}, Perplexity: {total_perplexity},"
                          f" BPC: {total_bpc}, Accuracy: {total_accuracy}, TimeFromStart: {time.time() - start_time}")
            print(f"Final testing stats | Loss: {total_loss}, Perplexity: {total_perplexity}, BPC: {total_bpc},"
                  f" Accuracy: {total_accuracy}, TimeSpent: {time.time() - start_time}")


if __name__ == '__main__':  # Main function
    TRAIN_DATA, VALID_DATA, TEST_DATA, vocabulary_size = load_data(data_set_name)  # Load data set

    # To see how it trains in small amounts (If it's usually in a 90%/5%/5% split, now it's in a 1%/1%/1% split)
    '''
    TRAIN_DATA = TRAIN_DATA[:len(TRAIN_DATA) // 90]
    VALID_DATA = VALID_DATA[:len(VALID_DATA) // 5]
    TEST_DATA = TEST_DATA[:len(TEST_DATA) // 5]
    '''

    # From which function/class we can get the model
    model_function = LMModel

    HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                             number_of_parameters=number_of_parameters,
                                             model_function=model_function)

    model = model_function(HIDDEN_UNITS)  # Create the model

    model.fit(TRAIN_DATA, VALID_DATA)  # Train the model (validating after each epoch)

    model.evaluate(TEST_DATA)  # Test the last saved model
