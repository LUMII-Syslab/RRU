import tensorflow as tf
import numpy as np
import time
from datetime import datetime  # We'll use this to dynamically generate training event names

import argparse

from sklearn.utils import shuffle

# Importing functions to load the data in the correct format
from imdb_utils import load_data

from imdb_utils import get_sequence_lengths

# Importing some utility functions that will help us with certain tasks
from utils import find_optimal_hidden_units
from utils import print_trainable_variables

# If you have many GPUs available, you can specify which one to use here (they are indexed from 0)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# You can check if this line makes it train faster, while still training correctly
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

# Choose your cell
cell_name = "RRU3"  # Here you can type in the name of the cell you want to use

# Maybe we can put these in a separate file called cells.py or something, and import it
output_size = None  # Most cells don't have an output size, so we by default set it as None
state_is_tuple = False  # We use variable length dynamic_rnn, so we need to know this if we want to use the code we have
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
vocabulary_size = 10000  # 24902 is the max (used to be 88583 for tfds)
max_sequence_length = 100  # [70-2697] words in tf.keras (used to be [6-2493] words in tdfs). None means max
batch_size = 64
num_epochs = 10
HIDDEN_UNITS = 128
number_of_parameters = 500000  # 500 K? That's what about 128 units gave us
number_of_layers = 2
embedding_size = 32
num_classes = 2
learning_rate = 0.001
shuffle_data = True
fixed_batch_size = False

ckpt_path = 'ckpt_imdb/'
log_path = 'logdir_imdb/'

# After how many steps should we send the data to TensorBoard (0 - don't log after any amount of steps)
log_after_this_many_steps = 0
assert log_after_this_many_steps >= 0, "Invalid value for variable log_after_this_many_steps, it must be >= 0!"
# After how many steps should we print the results of training/validating/testing (0 - don't print until the last step)
print_after_this_many_steps = 100
assert print_after_this_many_steps >= 0, "Invalid value for variable print_after_this_many_steps, it must be >= 0!"


current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = log_path + model_name + '/' + current_time


class IMDBModel:

    def __init__(self, hidden_units):

        print("\nBuilding Graph...\n")

        tf.reset_default_graph()  # Build the graph

        # Batch size list of integer sequences
        x = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name="x")
        # One hot labels for sentiment classification
        y = tf.placeholder(tf.int32, shape=[None, num_classes], name="y")
        # Bool value if we are in training or not
        training = tf.placeholder(tf.bool, name='training')
        # Batch size list of sequence lengths, so we can get variable sequence length rnn
        sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")

        # Cast our label to float32. Later it will be better when it does some math (?)
        y = tf.cast(y, tf.float32)

        # Instantiate our embedding matrix
        embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
                                name="word_embedding")

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

        # Extract the batch size - this allows for variable batch size
        current_batch_size = tf.shape(x)[0]

        # Create the initial state of zeros
        initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)

        # Wrap our cell in a dropout wrapper
        # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.85)

        # Value will have all the outputs, we will need just the last. _ contains hidden states between the steps
        if state_is_tuple:  # # BasicLSTMCell and Mogrifier LSTM needs this
            value, (_, state) = tf.nn.dynamic_rnn(cell,
                                                  embed_lookup,
                                                  initial_state=initial_state,
                                                  dtype=tf.float32,
                                                  sequence_length=sequence_length)
        else:
            value, state = tf.nn.dynamic_rnn(cell,
                                             embed_lookup,
                                             initial_state=initial_state,
                                             dtype=tf.float32,
                                             sequence_length=sequence_length)

        # Instantiate weights
        weight = tf.get_variable("weight", [hidden_units, num_classes])
        # Instantiate biases
        bias = tf.Variable(tf.constant(0.0, shape=[num_classes]))

        '''Non-variable sequence length dynamic.rnn'''
        # Hasn't been updated for layered cells (because I think we won't need this anymore)
        # value = tf.transpose(value, [1, 0, 2])  # After this it's max_time, batch_size, final_size
        # last = value[-1]  # Extract last output. Should be batch_size final_size
        '''Variable sequence length'''
        last = state[-1]  # Used to be just last = state, but we have layered cells now, so we take the last one

        prediction = tf.matmul(last, weight) + bias  # What we actually do is calculate the loss over the batch

        # predictions -        [1,1,0,0]
        # labels -             [1,0,0,1]
        # correct_prediction - [1,0,1,0]. We want accuracy over the batch, so we create a mean over it
        correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

        # Choices our model made
        # choice = tf.argmax(prediction, axis=1)

        # Calculate the loss given prediction and labels
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction, label_smoothing=0.1)
        tf.summary.scalar("loss", loss)

        # Declare our optimizer, we have to check which one works better.
        # Before: Adam gave better training accuracy and loss, but RMSProp gave better validation accuracy and loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

        # Expose symbols to class
        # Placeholders
        self.x = x
        self.y = y
        self.training = training
        self.sequence_length = sequence_length
        # Information you can get from this graph
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

        print_trainable_variables()

        print("\nGraph Built...\n")

    def fit(self, x_train, y_train):
        with tf.Session() as sess:
            print("|*|*|*|*|*| Starting training... |*|*|*|*|*|")

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Adding a writer so we can visualize accuracy and loss on TensorBoard
            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(output_path)
            train_writer.add_graph(sess.graph)

            for epoch in range(num_epochs):
                print(f"------ Epoch {epoch + 1} out of {num_epochs} ------")
                if epoch > 0:
                    data = list(zip(x_train, y_train))
                    if shuffle_data:
                        shuffle(data)
                    x_train, y_train = zip(*data)

                num_batches = len(x_train) // batch_size

                total_loss = 0
                total_accuracy = 0

                start_time = time.time()

                for i in range(num_batches):
                    if fixed_batch_size or i != num_batches - 1:
                        x_batch = x_train[i * batch_size: i * batch_size + batch_size]
                        y_batch = y_train[i * batch_size: i * batch_size + batch_size]
                    else:
                        x_batch = x_train[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                        y_batch = y_train[i * batch_size:]
                    sequence_lengths = get_sequence_lengths(x_batch)

                    feed_dict = {
                        self.x: x_batch,
                        self.y: y_batch,
                        self.training: True,
                        self.sequence_length: sequence_lengths
                    }

                    if log_after_this_many_steps != 0 and i % log_after_this_many_steps == 0:
                        s, _, l, a = sess.run([merged_summary, self.optimizer, self.loss, self.accuracy],
                                              feed_dict=feed_dict)

                        train_writer.add_summary(s, i + epoch * num_batches)
                    else:
                        _, l, a = sess.run([self.optimizer, self.loss, self.accuracy],
                                           feed_dict=feed_dict)

                    total_loss += l
                    if i == 0:
                        total_accuracy = a
                    else:
                        total_accuracy = (total_accuracy * i + a) / (i + 1)

                    if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0) or i == num_batches - 1:
                        print(f"Step {i + 1} of {num_batches} | Loss: {l}, Accuracy: {a}, TimeFromStart: {time.time() - start_time}")

                print(f"   Epoch {epoch + 1} | Loss: {total_loss}, Accuracy: {total_accuracy}, TimeSpent: {time.time() - start_time}")

                epoch_accuracy_summary = tf.Summary()
                epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=total_accuracy)
                train_writer.add_summary(epoch_accuracy_summary, epoch + 1)

                epoch_loss_summary = tf.Summary()
                epoch_loss_summary.value.add(tag='epoch_loss', simple_value=total_loss)
                train_writer.add_summary(epoch_loss_summary, epoch + 1)
            # Training ends here
            # Save checkpoint
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, ckpt_path + model_name + ".ckpt")

    def evaluate(self, x_test, y_test):
        with tf.Session() as sess:
            print("|*|*|*|*|*| Starting testing... |*|*|*|*|*|")

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Adding a writer so we can visualize accuracy and loss on TensorBoard
            test_writer = tf.summary.FileWriter(output_path + "/testing")
            test_writer.add_graph(sess.graph)

            # Restore session
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver = tf.compat.v1.train.Saver()
            # If there is a correct checkpoint at the path restore it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            num_batches = len(x_test) // batch_size

            total_loss = 0
            total_accuracy = 0

            start_time = time.time()

            for i in range(num_batches):
                if fixed_batch_size or i != num_batches - 1:
                    x_batch = x_test[i * batch_size: i * batch_size + batch_size]
                    y_batch = y_test[i * batch_size: i * batch_size + batch_size]
                else:
                    x_batch = x_test[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                    y_batch = y_test[i * batch_size:]
                sequence_lengths = get_sequence_lengths(x_batch)

                l, a = sess.run([self.loss, self.accuracy], feed_dict={self.x: x_batch,
                                                                       self.y: y_batch,
                                                                       self.training: False,
                                                                       self.sequence_length: sequence_lengths})

                total_loss += l
                if i == 0:
                    total_accuracy = a
                else:
                    total_accuracy = (total_accuracy * i + a) / (i + 1)

                if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0) or i == num_batches - 1:
                    print(f"Step {i + 1} of {num_batches} | Loss: {total_loss}, Accuracy: {total_accuracy}, TimeFromStart: {time.time() - start_time}")

            print(f"Final testing stats | Loss: {total_loss}, Accuracy: {total_accuracy}, TimeSpent: {time.time() - start_time}")

            # We add this to TensorBoard so we don't have to dig in console logs and nohups
            testing_accuracy_summary = tf.Summary()
            testing_accuracy_summary.value.add(tag='testing_accuracy', simple_value=total_accuracy)
            test_writer.add_summary(testing_accuracy_summary, 1)

            testing_loss_summary = tf.Summary()
            testing_loss_summary.value.add(tag='testing_loss', simple_value=total_loss)
            test_writer.add_summary(testing_loss_summary, 1)
            test_writer.flush()


def parse_args():  # Parse arguments. Currently not used, we will need to write it differently later.
    parser = argparse.ArgumentParser(description='Different RNN cell comparison for IMDB review sentiment analysis')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--train', action='store_true', help='train model')
    group.add_argument('-v', '--validate', action='store_true', help='validate model')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':  # Main function
    # ARGS = parse_args()  # We'll use this when we implement parse_args again

    # Load the IMDB data set
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, max_sequence_length = load_data(vocabulary_size, max_sequence_length)

    # From which function/class we can get the model
    model_function = IMDBModel

    HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                             number_of_parameters=number_of_parameters,
                                             model_function=model_function)

    model = model_function(HIDDEN_UNITS)  # Create the model

    model.fit(X_TRAIN, Y_TRAIN)

    model.evaluate(X_TEST, Y_TEST)
