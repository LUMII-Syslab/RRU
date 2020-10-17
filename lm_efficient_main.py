# If you have many GPUs available, you can specify which one to use here (they are indexed from 0)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# You can check if this line makes it train faster, while still training correctly
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

import tensorflow as tf
import numpy as np
import time
import math
from datetime import datetime  # We'll use this to dynamically generate training event names

from sklearn.utils import shuffle  # We'll use this to shuffle training data

# Importing some functions that will help us deal with the input data
from lm_efficient_utils import load_data
from lm_efficient_utils import get_window_indexes
from lm_efficient_utils import get_input_data_from_indexes

# Importing fancier optimizer(s)
# from RAdam import RAdamOptimizer
from adam_decay import AdamOptimizer_decay

# Choose your cell
cell_name = "RRU5"  # Here you can type in the name of the cell you want to use

# Maybe we can put these in a separate file called cells.py or something, and import it
output_size = None  # Most cells don't have an output size, so we by default set it as None
state_is_tuple = False  # So we can make the RNN cell stateful even for LSTM based cells such as LSTM and MogrifierLSTM
has_training_bool = False
if cell_name == "RRU1":  # ReZero version
    from RRUCell import RRUCell
    cell_fn = RRUCell
    output_size = 256
    has_training_bool = True
    model_name = 'rru_model'

elif cell_name == "RRU2":  # Gated version with 1 transformation
    from GatedRRUCell import RRUCell  # (I already have some results with this one)
    cell_fn = RRUCell
    has_training_bool = True
    model_name = 'grru1_model'

elif cell_name == "RRU3":  # Gated version with 2 transformations
    from GatedRRUCell2 import RRUCell
    cell_fn = RRUCell
    has_training_bool = True
    model_name = 'grru2_model'

elif cell_name == "RRU4":  # Gated version with 2 transformations and a separate output size
    from GatedRRUCell2_a import RRUCell
    cell_fn = RRUCell
    has_training_bool = True
    output_size = 256
    model_name = 'grru2a_model'

elif cell_name == "RRU5":  # Gated version with separate output size
    from GatedRRUCell_a import RRUCell
    cell_fn = RRUCell
    has_training_bool = True
    output_size = 256
    model_name = "grrua_model"  # We have hopes for this one

elif cell_name == "GRU":
    from GRUCell import GRUCell
    cell_fn = GRUCell
    model_name = 'gru_model'

elif cell_name == "LSTM":
    from BasicLSTMCell import BasicLSTMCell
    cell_fn = BasicLSTMCell
    state_is_tuple = True
    model_name = 'lstm_model'

elif cell_name == "MogrifierLSTM":  # Comment this out and you don't have to have dm-sonnet, etc. installed
    from tiled_lstm import TiledLSTMCell
    cell_fn = TiledLSTMCell
    state_is_tuple = True
    model_name = 'mogrifier_lstm_model'

else:
    raise ValueError(f"No such cell ('{cell_name}') has been implemented!")

# Hyperparameters
# Data parameters
data_set_name = "enwik8"  # "enwik8" | "text8" | "pennchar" | "penn" (which data set to test on)
vocabulary_size = None  # We will load this from a pickle file, so changing this here won't do a thing
window_size = 512  # Enwik8 – 512. Text8 512?, PTB word-level 70?, PTB character-level 150?
step_size = window_size // 2
batch_size = 64  # Enwik8 – 64. PTB character-level 128?
fixed_batch_size = False  # With this False it may run some batches on size [batch_size, 2 * batch_size)
continuous_batches = True  # Batches go continuously, this might give performance boost with a stateful RNN cell
shuffle_data = False  # Should we shuffle the samples?
# Training
num_epochs = 1000000  # We can code this to go infinity, but I see no point, we won't wait 1 million epochs anyway
break_epochs_no_gain = 3  # If validation BPC doesn't get lower, after how many epochs we should break (-1 -> disabled)
hidden_units = 1024  # This will only be used if the number_of_parameters is None or < 1
number_of_parameters = 24000000  # 24 million learnable parameters
embedding_size = 128
learning_rate = 0.001  # At 0,001 LSTM and GRU explodes a bit, and at 0.0001 Mogrifier LSTM can't learn, so 0,0005!
number_of_layers = 2
stateful = True  # Should the RNN cell be stateful? If True, you can modify it's zero_state_chance below.
zero_state_chance = 0.1  # Chance that zero_state is passed instead of last state (I don't know what value is best yet)

ckpt_path = 'ckpt_lm/'
log_path = 'logdir_lm/'

# After how many steps should we send the data to TensorBoard (0 – don't log after any amount of steps)
log_after_this_many_steps = 0
assert log_after_this_many_steps >= 0, "Invalid value for variable log_after_this_many_steps, it must be >= 0!"
# After how many steps should we print the results of training/validating/testing (0 – don't print until the last step)
print_after_this_many_steps = 100
assert print_after_this_many_steps >= 0, "Invalid value for variable print_after_this_many_steps, it must be >= 0!"

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = log_path + model_name + '/enwik8/' + current_time


class RNNLMModel:

    def __init__(self):

        print("\nBuilding Graph...\n")

        tf.reset_default_graph()  # Build the graph

        # Batch size list of integer sequences
        x = tf.placeholder(tf.int32, shape=[None, window_size], name="x")

        # Labels for word prediction
        y = tf.placeholder(tf.int64, shape=[None, window_size], name="y")

        # Bool value if we are in training or not
        training = tf.placeholder(tf.bool, name='training')

        # Instantiate our embedding matrix
        embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="word_embedding")

        # Lookup embeddings
        embed_lookup = tf.nn.embedding_lookup(embedding, x)

        # Create the RNN cell, corresponding to the one you chose, for example, RRU, GRU, LSTM, MogrifierLSTM
        # cell = cell_fn(hidden_units, training=training)  # Testing
        # Old 1 layer way
        # cell = cell_fn(hidden_units)
        # New 1+ layer way
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
        # Placeholders
        self.x = x
        self.y = y
        self.training = training
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

        trainable_variables = tf.trainable_variables()
        variables_total = 0
        for v in trainable_variables:
            variables_total += np.product(v.get_shape().as_list())
        print("Learnable parameters:", variables_total / 1000 / 1000, 'M', flush=True)  # Some divide it by 1024 twice

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
                        zero_state = np.zeros((len(x_batch), hidden_units))
                        if state_is_tuple:
                            state = []
                            state.append(zero_state)
                            state.append(zero_state)
                            zero_state = state
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
                            self.initial_state: state
                        }
                    else:
                        feed_dict = {
                            self.x: x_batch,
                            self.y: y_batch,
                            self.training: True
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
                            if fixed_batch_size or i != num_validation_batches - 1:  # The batch_size is fixed or it's not the last
                                x_batch = validation_indexes[i * batch_size: i * batch_size + batch_size]
                            else:
                                # Run the remaining sequences (that might not be exactly batch_size (they might be larger))
                                x_batch = validation_indexes[i * batch_size:]

                        if stateful and (np.random.uniform() < zero_state_chance or i == 0):
                            # state = np.zeros((number_of_layers, len(x_batch), hidden_units))
                            zero_state = np.zeros((len(x_batch), hidden_units))
                            if state_is_tuple:
                                state = []
                                state.append(zero_state)
                                state.append(zero_state)
                                zero_state = state
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
                                self.initial_state: state
                            }
                        else:
                            feed_dict = {
                                self.x: x_batch,
                                self.y: y_batch,
                                self.training: False
                            }

                        last_state, l, p, b, a = sess.run([self.state, self.loss, self.perplexity, self.bpc, self.accuracy],
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
            '''We used to save here – model saved after the last epoch'''
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
                    zero_state = np.zeros((len(x_batch), hidden_units))
                    if state_is_tuple:
                        state = []
                        state.append(zero_state)
                        state.append(zero_state)
                        zero_state = state
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
                        self.initial_state: state
                    }
                else:
                    feed_dict = {
                        self.x: x_batch,
                        self.y: y_batch,
                        self.training: False
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


def find_optimal_hidden_units():
    # Inspired from https://github.com/deepmind/lamb/blob/master/lamb/lamb_flags.py
    # They made a version that takes in a config file, which might be more useful to some people
    print(f"Searching for the largest possible hidden unit count"
          f",  which has <= {number_of_parameters} trainable parameters!")

    global hidden_units

    # If there wasn't given correct number of total parameters, then just use the given hidden units
    if number_of_parameters is None or number_of_parameters < 1:
        return hidden_units

    # If code goes this far, we don't care about the value in hidden_units variable anymore
    # , we will change it after it returns something

    def calculate_num_params_given_hidden_units(units):
        global hidden_units
        hidden_units = units

        # Before we used "test_model = RNNLMModel()" and in the end "del test_model", but it doesn't seem to help, but
        # If some time later you get some memory error, you can probably try this

        RNNLMModel()

        # Get the number of parameters in the current model
        trainable_variables = tf.trainable_variables()
        variable_count = 0
        for variable in trainable_variables:
            variable_count += np.product(variable.get_shape().as_list())

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


if __name__ == '__main__':  # Main function
    TRAIN_DATA, VALID_DATA, TEST_DATA, vocabulary_size = load_data(data_set_name)  # Load data set

    # To see how it trains in small amounts (If it's usually in a 90%/5%/5% split, now it's in a 1%/1%/1% split)
    '''
    TRAIN_DATA = TRAIN_DATA[:len(TRAIN_DATA) // 90]
    VALID_DATA = VALID_DATA[:len(VALID_DATA) // 5]
    TEST_DATA = TEST_DATA[:len(TEST_DATA) // 5]
    '''

    hidden_units = find_optimal_hidden_units()

    model = RNNLMModel()  # Create the model

    model.fit(TRAIN_DATA, VALID_DATA)  # Train the model (validating after each epoch)

    model.evaluate(TEST_DATA)  # Test the last saved model
