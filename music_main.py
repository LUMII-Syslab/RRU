import tensorflow as tf
import numpy as np
import time
from datetime import datetime  # We'll use this to dynamically generate training event names

from sklearn.utils import shuffle  # We'll use this to shuffle training data

# Importing some functions that will help us deal with the input data
from music_utils import load_data, split_data_in_parts

# Importing some utility functions that will help us with certain tasks
from utils import find_optimal_hidden_units, print_trainable_variables, get_batch, save_model, restore_model
from utils import print_trials_information, NetworkPrint

from cell_registry import get_cell_information

# Importing the necessary stuff for hyperparameter optimization
from hyperopt import hp, tpe, Trials, fmin

# Importing fancier optimizer(s)
from RAdam import RAdamOptimizer

# If you have many GPUs available, you can specify which one to use here (they are indexed from 0)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# You can check if this line makes it train faster, while still training correctly
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

# Choose your cell
cell_name = "RRU"  # Here you can type in the name of the cell you want to use

cell_fn, model_name, has_separate_output_size, _ = get_cell_information(cell_name)

# Hyperparameters
# Data parameters
# Choose on of "JSB Chorales" | "MuseData" | "Nottingham" | "Piano-midi.de" (which data set to test on)
data_set_name = "Nottingham"
vocabulary_size = None  # We will load this from a pickle file, so changing this here won't do a thing
window_size = 200  # If you have a lot of resources you can run this on full context size - 160/3780/1793/3623
step_size = window_size // 2
batch_size = 16  # Max batch_sizes: JSB Chorales 76; MuseData 124; Nottingham 170; Piano-midi.de 12
fixed_batch_size = False  # With this False it may run some batches on size [batch_size, 2 * batch_size)
shuffle_data = True  # Should we shuffle the samples?
# Training
num_epochs = 1000000  # We can code this to go infinity, but I see no point, we won't wait 1 million epochs anyway
break_epochs_no_gain = 7  # If validation BPC doesn't get lower, after how many epochs we should break (-1 -> disabled)
HIDDEN_UNITS = 128 * 3  # This will only be used if the number_of_parameters is None or < 1
number_of_parameters = 5000000  # 1 million learnable parameters
learning_rate = 0.001
number_of_layers = 1
do_hyperparameter_optimization = False
middle_layer_size_multiplier = 2
# gate_bias = 1
dropout_rate = 0.5
clip_gradients = True
clip_multiplier = 1.5  # This value matters only if clip_gradients = True

ckpt_path = 'ckpt_music/'
log_path = 'logdir_music/'

# After how many steps should we send the data to TensorBoard (0 - don't log after any amount of steps)
log_after_this_many_steps = 0
assert log_after_this_many_steps >= 0, "Invalid value for variable log_after_this_many_steps, it must be >= 0!"
# After how many steps should we print the results of training/validating/testing (0 - don't print until the last step)
print_after_this_many_steps = 1
assert print_after_this_many_steps >= 0, "Invalid value for variable print_after_this_many_steps, it must be >= 0!"

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = log_path + model_name + f'/{data_set_name}/' + current_time

# Don't print TensorFlow messages, that we don't need
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class MusicModel:

    def __init__(self, hidden_units):

        print("\nBuilding Graph...\n")

        tf.reset_default_graph()  # Build the graph

        # Batch size list of window size list of binary masked integer sequences
        x = tf.placeholder(tf.float32, shape=[None, window_size, vocabulary_size], name="x")

        # Batch size list of window size list of binary masked integer sequences
        y = tf.placeholder(tf.float32, shape=[None, window_size, vocabulary_size], name="y")

        # Bool value if we are in training or not
        training = tf.placeholder(tf.bool, name='training')

        # Batch size list of sequence multipliers, to get a fair loss
        sequence_length_matrix = tf.placeholder(tf.float32, [None, window_size], name="sequence_length_matrix")

        # Create the RNN cell, corresponding to the one you chose
        cells = []
        for _ in range(number_of_layers):
            if cell_name in ["RRU", "GRRUA"]:
                cell = cell_fn(hidden_units,
                               training=training,
                               output_size=output_size,
                               # gate_bias=gate_bias,
                               dropout_rate=dropout_rate,
                               middle_layer_size_multiplier=middle_layer_size_multiplier)
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

        # Optionally apply relu activation for RRU and GRRUA cells
        # value = tf.nn.relu(value)

        # Reshape outputs [batch_size, window_size, final_size] -> [batch_size x window_size, final_size]
        last = tf.reshape(value, shape=(-1, final_size))

        # Instantiate weights and biases
        weight = tf.get_variable("output", [final_size, vocabulary_size])
        bias = tf.Variable(tf.constant(0.0, shape=[vocabulary_size]))

        # Final form should be [batch_size x window_size, vocabulary_size]
        prediction = tf.matmul(last, weight) + bias  # What we actually do is calculate the loss over the batch
        # Reshape the predictions to match y dimensions
        # [batch_size x window_size, vocabulary_size] -> [batch_size, window_size, vocabulary_size]
        prediction = tf.reshape(prediction, shape=(-1, window_size, vocabulary_size))

        # Calculate NLL loss
        # After the next operation, shape will be [batch_size, window_size, vocabulary_size]
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
        # After the next operation, shape will be [batch_size, window_size]
        loss = tf.reduce_sum(loss, -1)
        # We nullify the padding loss and multiply the others, so we get a loss, which only counts in the un-padded data
        loss = loss * sequence_length_matrix
        # After the next operation, the shape will be [1]
        loss = tf.reduce_mean(loss)

        # Printing trainable variables which have "kernel" in their name
        decay_vars = [v for v in tf.trainable_variables() if 'kernel' in v.name]
        for c in decay_vars:
            print(c)

        # Declare our optimizer, we have to check which one works better.
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        optimizer = RAdamOptimizer(learning_rate=learning_rate,
                                   L2_decay=0.0,
                                   decay_vars=decay_vars,
                                   clip_gradients=clip_gradients, clip_multiplier=clip_multiplier).minimize(loss)
        # optimizer = RAdamOptimizer(learning_rate=learning_rate, L2_decay=0.0, epsilon=1e-8).minimize(loss)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

        # What to log to TensorBoard if a number was specified for "log_after_this_many_steps" variable
        tf.summary.scalar("loss", loss)

        # Expose symbols to class
        # Placeholders
        self.x = x
        self.y = y
        self.training = training
        self.sequence_length_matrix = sequence_length_matrix
        # Information you can get from this graph
        self.loss = loss
        # To call the optimization step (gradient descent)
        self.optimizer = optimizer

        print_trainable_variables()

        print("\nGraph Built...\n")

    def fit(self, train_data, valid_data=None):
        tf_config = tf.ConfigProto()
        # tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        sess = tf.Session(config=tf_config)
        NetworkPrint.training_start()
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        # Adding a writer so we can visualize accuracy and loss on TensorBoard
        merged_summary = tf.summary.merge_all()
        training_writer = tf.summary.FileWriter(output_path + "/training")
        training_writer.add_graph(sess.graph)

        x_train, y_train, sequence_lengths_train = split_data_in_parts(train_data, window_size, step_size,
                                                                       vocabulary_size)

        num_training_batches = len(x_train) // batch_size

        # Variables that help implement the early stopping if no validation loss decrease is observed
        epochs_no_gain = 0
        best_validation_loss = None

        for epoch in range(num_epochs):
            NetworkPrint.epoch_start(epoch + 1, num_epochs)

            if shuffle_data:
                x_train, y_train, sequence_lengths_train = shuffle(x_train, y_train, sequence_lengths_train)

            total_training_loss = 0

            start_time = time.time()

            for i in range(num_training_batches):
                x_batch = get_batch(x_train, i, batch_size, fixed_batch_size)
                y_batch = get_batch(y_train, i, batch_size, fixed_batch_size)
                sequence_length_batch = get_batch(sequence_lengths_train, i, batch_size, fixed_batch_size)

                sequence_length_matrix = create_sequence_length_matrix(len(sequence_length_batch),
                                                                       sequence_length_batch)

                feed_dict = {
                    self.x: x_batch,
                    self.y: y_batch,
                    self.training: True,
                    self.sequence_length_matrix: sequence_length_matrix
                }

                if log_after_this_many_steps != 0 and i % log_after_this_many_steps == 0:
                    s, _, loss = sess.run([merged_summary, self.optimizer, self.loss], feed_dict=feed_dict)

                    training_writer.add_summary(s, i + epoch * num_training_batches)
                else:
                    _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                total_training_loss += loss

                if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                        or i == num_training_batches - 1:
                    NetworkPrint.step_results(i + 1, num_training_batches, [["Loss", loss]], time.time() - start_time)

            average_training_loss = total_training_loss / num_training_batches
            NetworkPrint.epoch_end(epoch + 1, [["Average loss", average_training_loss]], time.time() - start_time)

            epoch_nll_summary = tf.Summary()
            epoch_nll_summary.value.add(tag='epoch_nll', simple_value=average_training_loss)
            training_writer.add_summary(epoch_nll_summary, epoch + 1)
            training_writer.flush()

            average_validation_loss = self.evaluate(valid_data, "validation", sess, epoch + 1)

            '''Here the training and validation epoch have been run, now get the lowest validation loss model'''
            # Check if validation loss was better
            if best_validation_loss is None or average_validation_loss < best_validation_loss:
                print(f"&&& New best validation loss - before: {best_validation_loss};"
                      f" after: {average_validation_loss} - saving model...")

                best_validation_loss = average_validation_loss

                # Save checkpoint
                save_model(sess, ckpt_path, model_name)

                epochs_no_gain = 0
            elif break_epochs_no_gain >= 1:  # Validation loss was worse. Check if break_epochs_no_gain is on
                epochs_no_gain += 1

                print(f"&&& No validation loss decrease for {epochs_no_gain} epochs,"
                      f" breaking at {break_epochs_no_gain} epochs.")

                if epochs_no_gain == break_epochs_no_gain:
                    print(f"&&& Maximum epochs without validation loss decrease reached, breaking...")
                    return

    def evaluate(self, data, mode="testing", session=None, iterator=1):
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

        x, y, sequence_lengths = split_data_in_parts(data, window_size, window_size, vocabulary_size)

        num_batches = len(x) // batch_size

        total_loss = 0

        start_time = time.time()

        for i in range(num_batches):
            x_batch = get_batch(x, i, batch_size, fixed_batch_size)
            y_batch = get_batch(y, i, batch_size, fixed_batch_size)
            sequence_length_batch = get_batch(sequence_lengths, i, batch_size, fixed_batch_size)

            sequence_length_matrix = create_sequence_length_matrix(len(sequence_length_batch),
                                                                   sequence_length_batch)

            feed_dict = {
                self.x: x_batch,
                self.y: y_batch,
                self.training: False,
                self.sequence_length_matrix: sequence_length_matrix
            }

            loss = sess.run(self.loss, feed_dict=feed_dict)

            total_loss += loss

            if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                    or i == num_batches - 1:
                NetworkPrint.step_results(i + 1, num_batches, [["Average loss", total_loss / (i + 1)]],
                                          time.time() - start_time)
        average_loss = total_loss / num_batches
        NetworkPrint.evaluation_end(mode, [["Average loss", average_loss]], time.time() - start_time)

        # We add this to TensorBoard so we don't have to dig in console logs and nohups
        loss_summary = tf.Summary()
        loss_summary.value.add(tag=f'{mode}_nll', simple_value=average_loss)
        writer.add_summary(loss_summary, iterator)
        writer.flush()

        return average_loss


def create_sequence_length_matrix(batch_dimension, sequence_lengths):
    matrix = np.zeros([batch_dimension, window_size])

    for i in range(batch_dimension):
        sequence_length = sequence_lengths[i]

        multiplier_for_the_non_padded = window_size / sequence_length

        for j in range(sequence_length):
            matrix[i][j] = multiplier_for_the_non_padded
        # And for the rest of the sequence leave the zeros

    return matrix


if __name__ == '__main__':  # Main function
    output_size = 256 if has_separate_output_size else None

    TRAINING_DATA, VALIDATION_DATA, TESTING_DATA, vocabulary_size = load_data(data_set_name)  # Load data set

    # From which function/class we can get the model
    model_function = MusicModel

    if not do_hyperparameter_optimization:
        HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                 number_of_parameters=number_of_parameters,
                                                 model_function=model_function)

        MODEL = model_function(HIDDEN_UNITS)  # Create the model

        MODEL.fit(TRAINING_DATA, VALIDATION_DATA)  # Train the model (validating after each epoch)

        MODEL.evaluate(TESTING_DATA)  # Test the last saved model
    else:
        times_to_evaluate = 100

        # What to do with hp.uniforms that need to be rounded
        round_uniform = ['num_params', 'out_size']

        space = [
            # hp.uniform
            hp.uniform('num_params', 1000000, 15000000),
            hp.uniform('out_size', 32, 256),
            hp.uniform('middle_multiplier', 0.5, 8),
            hp.uniform('drop_rate', 0.0, 0.9),
            # hp.uniform('gate_bias_value', -1, 3),
            # hp.loguniform
            hp.loguniform('lr', np.log(0.0004), np.log(0.004))
        ]

        def objective(num_params, out_size, middle_multiplier, drop_rate, lr):
            # For some values we need extra stuff
            num_params = round(num_params)
            out_size = round(out_size)

            # We'll optimize these parameters
            global learning_rate, number_of_parameters
            global output_size, middle_layer_size_multiplier, dropout_rate  # gate_bias
            learning_rate = lr
            number_of_parameters = num_params
            output_size = out_size
            middle_layer_size_multiplier = middle_multiplier
            dropout_rate = drop_rate
            # gate_bias = gate_bias_value

            # This might give some clues
            global output_path
            output_path = f"{log_path}{model_name}/{data_set_name}/{current_time}" \
                          f"/num_params{num_params}out_size{out_size}middle_multiplier{middle_multiplier}" \
                          f"lr{lr}dropout{drop_rate}"  # gate_bias{gate_bias}

            global HIDDEN_UNITS
            global model_function

            HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                     number_of_parameters=number_of_parameters,
                                                     model_function=model_function)

            model = model_function(HIDDEN_UNITS)  # Create the model

            model.fit(TRAINING_DATA, VALIDATION_DATA)  # Train the model (validating after each epoch)

            return model.evaluate(TESTING_DATA)  # Test the last saved model (it returns testing loss)

        # https://github.com/hyperopt/hyperopt/issues/129
        def objective2(args):
            return objective(*args)

        # Create the algorithm
        tpe_algo = tpe.suggest
        # Create trials object
        tpe_trials = Trials()

        # Run 2000 evals with the tpe algorithm
        tpe_best = fmin(fn=objective2, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=times_to_evaluate)

        print_trials_information(hyperopt_trials=tpe_trials,
                                 round_uniform=round_uniform,
                                 metric="NLL")
