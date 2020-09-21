import tensorflow as tf
import numpy as np
import time
import math
from datetime import datetime  # We'll use this to dynamically generate training event names

from sklearn.utils import shuffle

from lm_efficient_utils import load_data
from lm_efficient_utils import get_window_indexes
from lm_efficient_utils import get_input_data_from_indexes

'''Importing competitor cells'''
from tiled_lstm import TiledLSTMCell  # Comment this out and you don't have to have dm-sonnet, etc. installed
from BasicLSTMCell import BasicLSTMCell
from GRUCell import GRUCell

'''Importing different versions of our cell (Uncomment the one you want to use)'''
# from RRUCell import RRUCell
from GatedRRUCell import RRUCell  # (This currently is the best version)
# from GatedRRUCell2 import RRUCell

import os
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"  # Jāpārbauda vai ir ātrāk un vai trenējas korekti!

# Hyperparameters
data_set_name = "enwik8"  # "enwik8" | "text8" | "pennchar" | "penn" (which data set to test on)
vocabulary_size = None  # We will load this from a pickle file, so changing this here won't do a thing
window_size = 512  # Enwik8 – 512. Text8 512?, PTB word-level 70?, PTB character-level 150?
step_size = window_size // 2
batch_size = 64  # Enwik8 – 64. PTB character-level 128?
num_epochs = 1000000  # 5! We can code this to go infinity, but I see no point, we won't wait 1 million epochs anyway
break_epochs_no_gain = 3  # If validation bpc doesn't get lower, after how many epochs we should break (-1 -> disabled)
hidden_units = 256 * 7
number_of_parameters = 24000000  # 24 million
embedding_size = 256
learning_rate = 0.0005  # At 0,001 LSTM and GRU explodes a bit, and at 0.0001 Mogrifier LSTM can't learn, so 0,0005!
output_keep_prob = 0.9

ckpt_path = 'ckpt_lm/'
log_path = 'logdir_lm/'

# Uncomment the corresponding model name for the cell you are using
# model_name = 'lstm_model'  # BasicLSTMCell.py
model_name = 'gru_model'  # GRUCell.py
# model_name = 'rru_model'  # RRUCell.py
# model_name = 'grru1_model'  # GatedRRUCell.py (This currently is the best version)
# model_name = 'grru2_model'  # GatedRRUCell2.py
# model_name = 'mogrifier_lstm_model'  # tiled_lstm.py

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = log_path + model_name + '/enwik8/' + current_time


class RNN_LM_Model:

    def __init__(self):

        def __graph__():
            tf.reset_default_graph()  # Build the graph

            # Batch size list of integer sequences
            x = tf.placeholder(tf.int32, shape=[None, window_size], name="x")

            # Labels for word prediction
            y = tf.placeholder(tf.int64, shape=[None, window_size], name="y")

            # Output drop probability so we can pass different values depending on training / testing
            output_drop_prob = tf.placeholder(tf.float32, name='output_drop_prob')

            # Instantiate our embedding matrix
            embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
                                    name="word_embedding")

            # Lookup embeddings
            embed_lookup = tf.nn.embedding_lookup(embedding, x)

            # Create LSTM/GRU/RRU/MogrifierLSTM cell
            # cell = BasicLSTMCell(hidden_units)
            cell = GRUCell(hidden_units)
            # cell = RRUCell(hidden_units, dropout_rate=output_drop_prob)
            # cell = TiledLSTMCell(hidden_units, feature_mask_rank=79, feature_mask_rounds=6)

            # Extract the batch size - this allows for variable batch size
            current_batch_size = tf.shape(x)[0]

            # Create the initial state of zeros
            initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)
            # initial_state+=np.asarray([1.0, -1.0]*(initial_state.get_shape().as_list()[-1]//2))*0.1
            # initial_state += 0.1

            # Wrap our cell in a dropout wrapper
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.85)

            # Value will have all the outputs. State contains hidden states between the steps.
            # embed_lookup = tf.unstack(embed_lookup, axis=1)
            value, state = tf.nn.dynamic_rnn(cell,
                                             embed_lookup,
                                             initial_state=initial_state,
                                             dtype=tf.float32)

            ''' Uncomment these if you want a lot of extra data (it might take a lot of memory space)'''
            # Instantiate weights
            # gate_img = tf.expand_dims(value[0:1, :, :], -1)
            # tf.summary.image("mem", gate_img, max_outputs=16)
            # tf.summary.image("mem", tf.transpose(gate_img, [0, 2, 1, 3]), max_outputs=16)
            # tf.summary.histogram("s_mul", tf.sigmoid(cell.S_bias_variable)*1.5)
            # tf.summary.scalar("stateWeight", cell.prev_state_weight)
            # tf.summary.scalar("W_mul", cell._W_mul)

            weight = tf.get_variable("weight", [hidden_units, vocabulary_size])
            # Instantiate biases
            bias = tf.Variable(tf.constant(0.0, shape=[vocabulary_size]))

            # [batch_size, max_time, hidden_units] -> [batch_size x max_time, hidden_units]
            # value = tf.stack(value, axis=1)
            # value -= tf.reduce_mean(value, [-1], keepdims=True)
            # value = instance_norm(value)

            last = tf.reshape(value, shape=(-1, hidden_units))

            # [batch_size, window_size] -> [batch_size x window_size]
            labels = tf.reshape(y, [-1])

            # Final form should be [batch_size x max_time, vocabulary_size]
            prediction = tf.matmul(last, weight) + bias  # What we actually do is calculate the loss over the batch

            ''' Last half accuracy predictions '''
            half = window_size // 2
            half_last = tf.reshape(value[:, half:, :], shape=(-1, hidden_units))
            half_prediction = tf.matmul(half_last, weight) + bias
            half_y = y[:, half:]
            half_correct_prediction = tf.equal(tf.argmax(half_prediction, axis=1), tf.reshape(half_y, [-1]))
            half_accuracy = tf.reduce_mean(tf.cast(half_correct_prediction, tf.float32))
            tf.summary.scalar("half_accuracy", half_accuracy)
            ''' Full size accuracy predicions '''
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

            # Declare our optimizer, we have to check which one works better.
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

            # Expose symbols to class
            self.x = x
            self.y = y
            self.output_drop_prob = output_drop_prob
            self.loss = loss
            self.perplexity = perplexity
            self.bpc = bpc
            self.optimizer = optimizer
            self.accuracy = accuracy
            self.prediction = prediction
            self.correct_prediction = correct_prediction

            tvars = tf.trainable_variables()
            print(tvars)
            vsum = 0
            for v in tvars:
                vsum += np.product(v.get_shape().as_list())
            print("learnable parameters:", vsum / 1024 / 1024, 'M', flush=True)

        # Build graph
        print("\nBuilding Graph...\n")
        __graph__()
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

            indexes = get_window_indexes(len(train_data), window_size, step_size)

            num_batches = len(indexes) // batch_size

            # Variables that help implement the early stopping if no validation perplexity (in BPC) decrease is observed
            epochs_no_gain = 0
            best_validation_bpc = None

            for epoch in range(num_epochs):
                print(f"------ Epoch {epoch + 1} out of {num_epochs} ------")

                indexes = shuffle(indexes)

                total_loss = 0
                total_accuracy = 0
                total_bpc = 0
                total_perplexity = 0

                start_time = time.time()

                for i in range(num_batches):
                    if i != num_batches - 1:
                        x_batch = indexes[i * batch_size: i * batch_size + batch_size]
                    else:
                        x_batch = indexes[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                    # Now we have batch of integers to look in text from
                    x_batch, y_batch = get_input_data_from_indexes(train_data, x_batch, window_size)

                    s, _, l, p, b, a = sess.run([merged_summary,
                                                 self.optimizer, self.loss, self.perplexity, self.bpc, self.accuracy],
                                                feed_dict={self.x: x_batch,
                                                self.y: y_batch,
                                                self.output_drop_prob: 1 - output_keep_prob})

                    # train_writer.add_summary(s, i + epoch * num_batches)
                    total_loss += l
                    if i == 0:
                        total_accuracy = a
                        total_bpc = b
                        total_perplexity = p
                    else:
                        total_accuracy = (total_accuracy * i + a) / (i + 1)
                        total_bpc = (total_bpc * i + b) / (i + 1)
                        total_perplexity = (total_perplexity * i + p) / (i + 1)

                    if i > 0 and ((i + 1) % 100 == 0 or i == num_batches - 1):
                        print(f"Step {i + 1} of {num_batches} | Loss: {l}, Perplexity: {p}, BPC: {b}, Accuracy: {a}, TimeFromStart: {time.time() - start_time}")

                print(f"   Epoch {epoch + 1} | Loss: {total_loss}, Perplexity: {total_perplexity}, BPC: {total_bpc}, Accuracy: {total_accuracy}, TimeSpent: {time.time() - start_time}")

                epoch_accuracy_summary = tf.Summary()
                epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=total_accuracy)
                train_writer.add_summary(epoch_accuracy_summary, epoch + 1)

                epoch_bpc_summary = tf.Summary()
                epoch_bpc_summary.value.add(tag='epoch_bpc', simple_value=total_bpc)
                train_writer.add_summary(epoch_bpc_summary, epoch + 1)

                if valid_data is not None:
                    print(f"------ Starting validation for epoch {epoch + 1} out of {num_epochs}... ------")

                    # I think we need to validate and test with step_size = step_size, right? maybe 1
                    validation_indexes = get_window_indexes(len(valid_data), window_size, step_size)

                    num_validation_batches = len(validation_indexes) // batch_size

                    total_val_loss = 0
                    total_val_accuracy = 0
                    total_val_bpc = 0
                    total_val_perplexity = 0

                    start_time = time.time()

                    for i in range(num_validation_batches):
                        if i != num_validation_batches - 1:
                            x_batch = validation_indexes[i * batch_size: i * batch_size + batch_size]
                        else:
                            # Run the remaining sequences (that aren't full batch_size)
                            x_batch = validation_indexes[i * batch_size:]
                        # Now we have batch of integers to look in text from
                        x_batch, y_batch = get_input_data_from_indexes(valid_data, x_batch, window_size)

                        l, p, b, a = sess.run([self.loss, self.perplexity, self.bpc, self.accuracy],
                                              feed_dict={self.x: x_batch,
                                                         self.y: y_batch,
                                                         self.output_drop_prob: 0.})
                        total_val_loss += l
                        if i == 0:
                            total_val_accuracy = a
                            total_val_bpc = b
                            total_val_perplexity = p
                        else:
                            total_val_accuracy = (total_val_accuracy * i + a) / (i + 1)
                            total_val_bpc = (total_val_bpc * i + b) / (i + 1)
                            total_val_perplexity = (total_val_perplexity * i + p) / (i + 1)

                        if i > 0 and ((i + 1) % 100 == 0 or i == num_validation_batches - 1):
                            print(f"Step {i + 1} of {num_validation_batches} | Loss: {total_val_loss}, Perplexity: {total_val_perplexity}, BPC: {total_val_bpc}, Accuracy: {total_val_accuracy}, TimeFromStart: {time.time() - start_time}")
                    print(f"Final validation stats | Loss: {total_val_loss}, Perplexity: {total_val_perplexity}, BPC: {total_val_bpc}, Accuracy: {total_val_accuracy}, TimeSpent: {time.time() - start_time}")

                    epoch_accuracy_summary = tf.Summary()
                    epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=total_val_accuracy)
                    validation_writer.add_summary(epoch_accuracy_summary, epoch + 1)

                    epoch_bpc_summary = tf.Summary()
                    epoch_bpc_summary.value.add(tag='epoch_bpc', simple_value=total_val_bpc)
                    validation_writer.add_summary(epoch_bpc_summary, epoch + 1)

                    '''Here the training and validation epoch have been run, now get the lowest validation bpc model'''
                    # Check if validation perplexity was better
                    if best_validation_bpc is None or total_val_bpc < best_validation_bpc:
                        print(f"&&& New best validation bpc - before: {best_validation_bpc}; after: {total_val_bpc} - saving model...")

                        best_validation_bpc = total_val_bpc

                        # Save checkpoint
                        saver = tf.compat.v1.train.Saver()
                        saver.save(sess, ckpt_path + model_name + ".ckpt")

                        epochs_no_gain = 0
                    elif break_epochs_no_gain >= 1:  # Validation BPC was worse. Check if break_epochs_no_gain is on
                        epochs_no_gain += 1
                        print(f"&&& No validation perplexity decrease for {epochs_no_gain} epochs, breaking at {break_epochs_no_gain} epochs.")
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

            for i in range(num_batches):
                if i != num_batches - 1:
                    x_batch = indexes[i * batch_size: i * batch_size + batch_size]
                else:
                    x_batch = indexes[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                # Now we have batch of integers to look in text from
                x_batch, y_batch = get_input_data_from_indexes(data, x_batch, window_size)

                l, p, b, a = sess.run([self.loss, self.perplexity, self.bpc, self.accuracy],
                                      feed_dict={self.x: x_batch,
                                                 self.y: y_batch,
                                                 self.output_drop_prob: 0.})
                total_loss += l
                if i == 0:
                    total_accuracy = a
                    total_bpc = b
                    total_perplexity = p
                else:
                    total_accuracy = (total_accuracy * i + a) / (i + 1)
                    total_bpc = (total_bpc * i + b) / (i + 1)
                    total_perplexity = (total_perplexity * i + p) / (i + 1)

                if i > 0 and ((i + 1) % 100 == 0 or i == num_batches - 1):
                    print(f"Step {i + 1} of {num_batches} | Loss: {total_loss}, Perplexity: {total_perplexity} BPC: {total_bpc}, Accuracy: {total_accuracy}, TimeFromStart: {time.time() - start_time}")
            print(f"Final testing stats | Loss: {total_loss}, Perplexity: {total_perplexity}, BPC: {total_bpc}, Accuracy: {total_accuracy}, TimeSpent: {time.time() - start_time}")


def find_optimal_hidden_units():
    # Inspired from https://github.com/deepmind/lamb/blob/master/lamb/lamb_flags.py
    # They made a version that takes in a config file, which might be more useful to some people
    print(f"Searching for the largest possible hidden unit count"
          f",  which has <= {number_of_parameters} trainable parameters!")

    global hidden_units

    # If there wasn't given correct number of total parameters, then just use the given hidden units
    if number_of_parameters < 1:
        return hidden_units

    # If code goes this far, we don't care about the value in hidden_units variable anymore
    # , we will change it after it returns something

    def calculate_num_params_given_hidden_units(units):
        global hidden_units
        hidden_units = units
        test_model = RNN_LM_Model()

        # Get the number of paramaters in the current model
        trainable_variables = tf.trainable_variables()
        variable_count = 0
        for variable in trainable_variables:
            variable_count += np.product(variable.get_shape().as_list())

        # These don't seem to be necessary, but they might help if there are some problems with memory
        tf.keras.backend.clear_session()
        del test_model

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


if __name__ == '__main__':  # Main function
    TRAIN_DATA, VALID_DATA, TEST_DATA, vocabulary_size = load_data(data_set_name)  # Load data set

    # To see how it trains in small amounts
    '''
    TRAIN_DATA = TRAIN_DATA[:len(TRAIN_DATA) // 90]
    VALID_DATA = VALID_DATA[:len(VALID_DATA) // 5]
    TEST_DATA = TEST_DATA[:len(TEST_DATA) // 5]
    '''

    hidden_units = find_optimal_hidden_units()

    model = RNN_LM_Model()  # Create the model

    model.fit(TRAIN_DATA, VALID_DATA)  # Train the model (validating after each epoch)

    model.evaluate(TEST_DATA)  # Test the model
