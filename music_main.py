import tensorflow as tf
import numpy as np
import time
import math
from datetime import datetime  # We'll use this to dynamically generate training event names

from sklearn.utils import shuffle  # We'll use this to shuffle training data

# Importing some functions that will help us deal with the input data
from music_utils import load_data
from music_utils import split_data_in_parts

# Importing fancier optimizer(s)
# from RAdam import RAdamOptimizer
from adam_decay import AdamOptimizer_decay

# If you have many GPUs available, you can specify which one to use here (they are indexed from 0)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# You can check if this line makes it train faster, while still training correctly
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

# Choose your cell
cell_name = "GRU"  # Here you can type in the name of the cell you want to use

# Maybe we can put these in a separate file called cells.py or something, and import it
output_size = None  # Most cells don't have an output size, so we by default set it as None
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
    model_name = 'grru1_model'

elif cell_name == "RRU3":  # Gated version with 2 transformations
    from cells.GatedRRUCell2 import RRUCell
    cell_fn = RRUCell
    has_training_bool = True
    model_name = 'grru2_model'

elif cell_name == "RRU4":  # Gated version with 2 transformations and a separate output size
    from cells.GatedRRUCell2_a import RRUCell
    cell_fn = RRUCell
    has_training_bool = True
    output_size = 256
    model_name = 'grru2a_model'

elif cell_name == "RRU5":  # Gated version with separate output size
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
    model_name = 'lstm_model'

elif cell_name == "MogrifierLSTM":  # Comment this out and you don't have to have dm-sonnet, etc. installed
    from cells.MogrifierLSTMCell import MogrifierLSTMCell
    cell_fn = MogrifierLSTMCell
    model_name = 'mogrifier_lstm_model'

else:
    raise ValueError(f"No such cell ('{cell_name}') has been implemented!")

# Hyperparameters
# Data parameters
# Choose on of "JSB Chorales" | "MuseData" | "Nottingham" | "Piano-midi.de" (which data set to test on)
data_set_name = "Nottingham"
vocabulary_size = None  # We will load this from a pickle file, so changing this here won't do a thing
window_size = 200  # If you have a lot of resources you can run this on full context size – 160/3780/1793/3623
step_size = window_size // 2
batch_size = 16  # 64
fixed_batch_size = False  # With this False it may run some batches on size [batch_size, 2 * batch_size)
shuffle_data = True  # Should we shuffle the samples?
# Training
num_epochs = 1000000  # We can code this to go infinity, but I see no point, we won't wait 1 million epochs anyway
break_epochs_no_gain = 3  # If validation BPC doesn't get lower, after how many epochs we should break (-1 -> disabled)
hidden_units = 128 * 3  # This will only be used if the number_of_parameters is None or < 1
number_of_parameters = 1000000  # 1 million learnable parameters
learning_rate = 0.001
number_of_layers = 2

ckpt_path = 'ckpt_music/'
log_path = 'logdir_music/'

# After how many steps should we send the data to TensorBoard (0 – don't log after any amount of steps)
log_after_this_many_steps = 0
assert log_after_this_many_steps >= 0, "Invalid value for variable log_after_this_many_steps, it must be >= 0!"
# After how many steps should we print the results of training/validating/testing (0 – don't print until the last step)
print_after_this_many_steps = 10
assert print_after_this_many_steps >= 0, "Invalid value for variable print_after_this_many_steps, it must be >= 0!"

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = log_path + model_name + f'/{data_set_name}/' + current_time


class MusicModel:

    def __init__(self):

        print("\nBuilding Graph...\n")

        tf.reset_default_graph()  # Build the graph

        # Batch size list of window size list of binary masked integer sequences
        x = tf.placeholder(tf.float32, shape=[None, window_size, vocabulary_size], name="x")

        # Batch size list of window size list of binary masked integer sequences
        y = tf.placeholder(tf.float32, shape=[None, window_size, vocabulary_size], name="y")

        # Bool value if we are in training or not
        training = tf.placeholder(tf.bool, name='training')

        # Create the RNN cell, corresponding to the one you chose
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

        # Create the initial state and initialize at the cell's zero state
        initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)

        # Value will have all the outputs. State contains hidden states between the steps.
        value, state = tf.nn.dynamic_rnn(cell,
                                         x,
                                         initial_state=initial_state,
                                         dtype=tf.float32)

        # Apply sigmoid activation function
        value = tf.nn.sigmoid(value)

        # Reshape outputs [batch_size, window_size, final_size] -> [batch_size x window_size, final_size]
        last = tf.reshape(value, shape=(-1, final_size))

        # Instantiate weights and biases
        weight = tf.get_variable("output", [final_size, vocabulary_size])
        bias = tf.Variable(tf.constant(0.0, shape=[vocabulary_size]))

        # Final form should be [batch_size x window_size, vocabulary_size]
        prediction = tf.matmul(last, weight) + bias  # What we actually do is calculate the loss over the batch

        # [batch_size, window_size, vocabulary_size] -> [batch_size x window_size, vocabulary_size]
        labels = tf.reshape(y, shape=(-1, vocabulary_size))

        # Calculate NLL loss
        # loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=prediction, multi_class_labels=labels))
        # maybe this will be more correct
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=labels)
        # loss = tf.losses.sigmoid_cross_entropy(logits=prediction, multi_class_labels=labels)
        # loss = tf.reshape(loss, shape=(current_batch_size, window_size, -1))
        loss = tf.reduce_sum(loss, -1)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", loss)

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
        # Information you can get from this graph
        self.loss = loss
        self.optimizer = optimizer
        self.prediction = prediction

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

            x_train, y_train = split_data_in_parts(train_data, window_size, step_size, vocabulary_size)

            num_batches = len(x_train) // batch_size

            # Variables that help implement the early stopping if no validation loss decrease is observed
            epochs_no_gain = 0
            best_validation_loss = None

            for epoch in range(num_epochs):
                print(f"------ Epoch {epoch + 1} out of {num_epochs} ------")

                if shuffle_data:
                    x_train, y_train = shuffle(x_train, y_train)  # Check if it this shuffles correctly maybe

                total_loss = 0

                start_time = time.time()

                for i in range(num_batches):
                    if fixed_batch_size or i != num_batches - 1:  # The batch_size is fixed or it's not the last
                        x_batch = x_train[i * batch_size: i * batch_size + batch_size]
                        y_batch = y_train[i * batch_size: i * batch_size + batch_size]
                    else:
                        # Run the remaining sequences (that might not be exactly batch_size (they might be larger))
                        x_batch = x_train[i * batch_size:]
                        y_batch = y_train[i * batch_size:]

                    feed_dict = {
                        self.x: x_batch,
                        self.y: y_batch,
                        self.training: True
                    }

                    if log_after_this_many_steps != 0 and i % log_after_this_many_steps == 0:
                        s, _, l = sess.run([merged_summary, self.optimizer, self.loss], feed_dict=feed_dict)

                        train_writer.add_summary(s, i + epoch * num_batches)
                    else:
                        _, l = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                    total_loss += l

                    if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                            or i == num_batches - 1:
                        print(f"Step {i + 1} of {num_batches} | Loss: {l}, TimeFromStart: {time.time() - start_time}")

                average_loss = total_loss / num_batches
                print(f"   Epoch {epoch + 1} | Average loss: {average_loss}, TimeSpent: {time.time() - start_time}")

                epoch_nll_summary = tf.Summary()
                epoch_nll_summary.value.add(tag='epoch_nll', simple_value=average_loss)
                train_writer.add_summary(epoch_nll_summary, epoch + 1)

                if valid_data is not None:
                    print(f"------ Starting validation for epoch {epoch + 1} out of {num_epochs}... ------")

                    x_valid, y_valid = split_data_in_parts(valid_data, window_size, step_size, vocabulary_size)

                    num_validation_batches = len(x_valid) // batch_size

                    total_val_loss = 0

                    start_time = time.time()

                    for i in range(num_validation_batches):
                        if fixed_batch_size or i != num_validation_batches - 1:  # The batch_size is fixed or it's not the last
                            x_batch = x_valid[i * batch_size: i * batch_size + batch_size]
                            y_batch = y_valid[i * batch_size: i * batch_size + batch_size]
                        else:
                            # Run the remaining sequences (that might not be exactly batch_size (they might be larger))
                            x_batch = x_valid[i * batch_size:]
                            y_batch = y_valid[i * batch_size:]

                        feed_dict = {
                            self.x: x_batch,
                            self.y: y_batch,
                            self.training: False
                        }

                        l = sess.run(self.loss, feed_dict=feed_dict)

                        total_val_loss += l

                        if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                                or i == num_validation_batches - 1:
                            print(f"Step {i + 1} of {num_validation_batches} | "
                                  f"Average loss: {total_val_loss / (i + 1)}, "
                                  f"TimeFromStart: {time.time() - start_time}")

                    average_val_loss = total_val_loss / num_validation_batches
                    print(f"Final validation stats | Average loss: {average_val_loss}, "
                          f"TimeSpent: {time.time() - start_time}")

                    epoch_nll_summary = tf.Summary()
                    epoch_nll_summary.value.add(tag='epoch_nll', simple_value=average_val_loss)
                    validation_writer.add_summary(epoch_nll_summary, epoch + 1)
                    validation_writer.flush()

                    '''Here the training and validation epoch have been run, now get the lowest validation loss model'''
                    # Check if validation loss was better
                    if best_validation_loss is None or average_val_loss < best_validation_loss:
                        print(f"&&& New best validation loss - before: {best_validation_loss};"
                              f" after: {average_val_loss} - saving model...")

                        best_validation_loss = average_val_loss

                        # Save checkpoint
                        saver = tf.compat.v1.train.Saver()
                        saver.save(sess, ckpt_path + model_name + ".ckpt")

                        epochs_no_gain = 0
                    elif break_epochs_no_gain >= 1:  # Validation loss was worse. Check if break_epochs_no_gain is on
                        epochs_no_gain += 1

                        print(f"&&& No validation loss decrease for {epochs_no_gain} epochs,"
                              f" breaking at {break_epochs_no_gain} epochs.")

                        if epochs_no_gain == break_epochs_no_gain:
                            print(f"&&& Maximum epochs without validation loss decrease reached, breaking...")
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

            x_test, y_test = split_data_in_parts(data, window_size, step_size, vocabulary_size)

            # Restore session
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver = tf.compat.v1.train.Saver()
            # If there is a correct checkpoint at the path restore it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            num_batches = len(x_test) // batch_size

            total_loss = 0

            start_time = time.time()

            for i in range(num_batches):

                if fixed_batch_size or i != num_batches - 1:  # The batch_size is fixed or it's not the last
                    x_batch = x_test[i * batch_size: i * batch_size + batch_size]
                    y_batch = y_test[i * batch_size: i * batch_size + batch_size]
                else:
                    # Run the remaining sequences (that might not be exactly batch_size (they might be larger))
                    x_batch = x_test[i * batch_size:]
                    y_batch = y_test[i * batch_size:]

                feed_dict = {
                    self.x: x_batch,
                    self.y: y_batch,
                    self.training: False
                }

                l = sess.run(self.loss, feed_dict=feed_dict)

                total_loss += l

                if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                        or i == num_batches - 1:
                    print(f"Step {i + 1} of {num_batches} | Average loss: {total_loss / (i + 1)},"
                          f" TimeFromStart: {time.time() - start_time}")
            average_loss = total_loss / num_batches
            print(f"Final testing stats | Average loss: {average_loss}, TimeSpent: {time.time() - start_time}")


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

        # Before we used "test_model = MusicModel()" and in the end "del test_model", but it doesn't seem to help, but
        # If some time later you get some memory error, you can probably try this

        MusicModel()

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

    model = MusicModel()  # Create the model

    model.fit(TRAIN_DATA, VALID_DATA)  # Train the model (validating after each epoch)

    model.evaluate(TEST_DATA)  # Test the last saved model
