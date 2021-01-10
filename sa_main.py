# This file implements the main functions for running the sentiment analysis task, this includes: model creation; model
# training, model testing and the main function that controls all the flow

# Importing the machine learning framework
import tensorflow as tf

# We'll use this to measure time spent in training and testing
import time

# We'll use this to dynamically generate training event names (so each run has different name)
from datetime import datetime

# We'll use this to shuffle training data
from sklearn.utils import shuffle

# Importing some functions that will help us deal with the input data
from sa_utils import load_data, get_sequence_lengths

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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# You can check if this line makes it train faster, while still training correctly
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

# Hyperparameters
# 1. Data parameters
vocabulary_size = 10000  # 24902 is the max (used to be 88583 for tfds)
# How many time steps we will unroll in RNN (IMDB has sentences with [70-2697] words) (None means max)
max_sequence_length = 500  # int, >= 1
# How many data samples we feed in a single time (IMDB maximum batch_size is 25000)
batch_size = 64  # int, >= 1
# # Can some of the batches be with size [batch_size, 2 * batch_size) (So we don't have as much left over data)
fixed_batch_size = False  # bool
# Should we shuffle the samples?
shuffle_data = True  # bool
# 2. Model parameters
# Name of the cell you want to test
cell_name = "RRU"  # string, one of these ["RRU", "GRRUA", "GRU", "LSTM", "MogrifierLSTM"]
# Number of hidden units (This will only be used if the number_of_parameters is None or < 1)
HIDDEN_UNITS = 128  # int, >= 1 (Probably way more than 1)
# Number of maximum allowed trainable parameters
number_of_parameters = 2000000  # int, >= 1 (Probably way more than 1)
# With what learning rate we optimize the model
learning_rate = 0.001  # float, > 0
# How many RNN layers should we have
number_of_layers = 2  # int, >= 1
# What should be the output size for the cells that have a separate output_size?
output_size = 256  # int, >= 1 (Probably way more than 1)
# Should we clip the gradients?
clip_gradients = True  # bool,
# If clip_gradients = True, with what multiplier should we clip them?
clip_multiplier = 5.  # float
# What should be the embedding size for the data (how many dimensions)?
embedding_size = 32  # int, >= 1
# 3. Training/testing process parameters
# Choose your cell
# How many epochs should we run?
num_epochs = 10  # int, >= 1
# Should we do hyperparameter optimization?
do_hyperparameter_optimization = False  # bool
# How many runs should we run hyperparameter optimization
optimization_runs = 100  # int, >= 1
# Path, where we will save the model for further evaluating
ckpt_path = 'ckpt_sa/'  # string
# Path, where we will store ours logs
log_path = 'logdir_sa/'  # string
# After how many steps should we send the data to TensorBoard (0 - don't log after any amount of steps)
log_after_this_many_steps = 0  # integer, >= 0
# After how many steps should we print the results of training/validating/testing (0 - don't print until the last step)
print_after_this_many_steps = 1  # integer, >= 0

# There are 2 classes for sentiment analysis for the "IMDB" data set
num_classes = 2

# Get information about the picked cell
cell_fn, model_name, has_separate_output_size, state_is_tuple = get_cell_information(cell_name)

# If the picked cell doesn't have a separate output size, we set is as None, so we can later process that
if not has_separate_output_size:
    output_size = None

# Calculating the path, in which we will store the logs
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")  # We put current time in the name, so it is unique in each run
output_path = log_path + model_name + '/IMDB/' + current_time  # IMDB while there is only 1 data set

# Don't print TensorFlow messages, that we don't need
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Class for solving sentiment analysis modeling tasks. You can create a model, train it and test it
class SentimentAnalysisModel:

    def __init__(self, hidden_units):
        """
            This function (also a constructor) creates a machine learning model for solving the sentiment analysis task.

            Input:
                hidden_units: int, the amount of hidden units to use for the RNN cell(s).
        """

        print("\nBuilding Graph...\n")

        # Build the graph
        tf.reset_default_graph()

        # Batch size list of integer sequences
        x = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name="x")

        # One hot labels for sentiment classification
        y = tf.placeholder(tf.int32, shape=[None, num_classes], name="y")

        # Bool value that tells us, whether or not are we in training
        training = tf.placeholder(tf.bool, name='training')

        # Batch size list of sequence lengths, so we can get variable length sequence RNN
        sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")

        # Cast our label to float32, it's more effective when doing some "math"
        y = tf.cast(y, tf.float32)

        # Instantiate our embedding matrix
        embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="word_embedding")

        # Lookup embeddings
        embed_lookup = tf.nn.embedding_lookup(embedding, x)

        # Create the RNN cell, corresponding to the one you chose, for example, RRU, GRU, LSTM, MogrifierLSTM
        cells = []
        for _ in range(number_of_layers):
            if cell_name in ["RRU", "GRRUA"]:
                cell = cell_fn(hidden_units, training=training, output_size=output_size)
            else:
                cell = cell_fn(hidden_units)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        # Extract the batch size - this allows for variable batch size
        current_batch_size = tf.shape(x)[0]

        # Create the initial state of zeros
        initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)

        # Value will have all the outputs.
        # State contains the final state.
        value, state = tf.nn.dynamic_rnn(cell,
                                         embed_lookup,
                                         initial_state=initial_state,
                                         dtype=tf.float32,
                                         sequence_length=sequence_length)

        if state_is_tuple:  # BasicLSTMCell and Mogrifier LSTM needs this
            state = state[-1]  # Last cell
            (_, state) = state  # We need only the second value from the tuple
            last = state
        else:
            # We take the last cell's state (for single cell networks, we could do just last = state)
            last = state[-1]

        # Instantiate weights
        weight = tf.get_variable("weight", [hidden_units, num_classes])

        # Instantiate biases
        bias = tf.Variable(tf.constant(0.0, shape=[num_classes]))

        # We calculate our predictions for each sample (we'll use this to calculate loss over the batch as well)
        prediction = tf.matmul(last, weight) + bias

        # We calculate on which samples were we correct
        # predictions        [1,1,0,0]
        # labels             [1,0,0,1]
        # correct_prediction [1,0,1,0]
        correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))

        # We want accuracy over the batch, so we create a mean over it
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Calculate the loss given prediction and labels
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction, label_smoothing=0.1)

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
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("loss", loss)

        # Expose symbols to class
        # Placeholders
        self.x = x
        self.y = y
        self.training = training
        self.sequence_length = sequence_length
        # Information you can get from this graph
        self.loss = loss
        self.accuracy = accuracy
        # To call the optimization step (gradient descent)
        self.optimizer = optimizer

        print_trainable_variables()

        print("\nGraph Built...\n")

    def fit(self, x_train, y_train):
        """
            This function trains the model using the training data passed.

            Input:
                x_train: list, a list of input sequences to be fed in the model;
                y_train: list, a list of input sequences to be fed in the model.
        """

        sess = tf.Session()

        # We print that the training has started
        NetworkPrint.training_start()

        # We initialize the variables
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        # Adding a writer so we can visualize accuracy and loss on TensorBoard
        merged_summary = tf.summary.merge_all()
        training_writer = tf.summary.FileWriter(output_path + "/training")
        # Adding session graph to the writer, so we can look at it, if we want, in TensorBoard
        training_writer.add_graph(sess.graph)

        # We calculate the number of batches
        num_batches = len(x_train) // batch_size

        for epoch in range(num_epochs):
            # Print that the epoch has started
            NetworkPrint.epoch_start(epoch + 1, num_epochs)

            # If necessary, shuffle data
            if shuffle_data:
                x_train, y_train = shuffle(x_train, y_train)

            total_loss = 0
            total_accuracy = 0

            start_time = time.time()

            for i in range(num_batches):
                # Get a batches of data
                x_batch = get_batch(x_train, i, batch_size, fixed_batch_size)
                y_batch = get_batch(y_train, i, batch_size, fixed_batch_size)

                # Get the sequence lengths of the input sequences
                sequence_lengths = get_sequence_lengths(x_batch)

                feed_dict = {
                    self.x: x_batch,
                    self.y: y_batch,
                    self.training: True,
                    self.sequence_length: sequence_lengths
                }

                # If we need to log this batch, we also add summary to the sess.run
                if log_after_this_many_steps != 0 and i % log_after_this_many_steps == 0:
                    s, _, l, a = sess.run([merged_summary, self.optimizer, self.loss, self.accuracy],
                                          feed_dict=feed_dict)

                    # Adding the summary to TensorBoard
                    training_writer.add_summary(s, i + epoch * num_batches)
                else:
                    _, l, a = sess.run([self.optimizer, self.loss, self.accuracy],
                                       feed_dict=feed_dict)

                total_loss += l
                total_accuracy += a

                # Print the batch results if it's the last batch or if step printing is turned on, and this is the step
                # to print in
                if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                        or i == num_batches - 1:
                    NetworkPrint.step_results(i + 1, num_batches, [["Loss", l], ["Accuracy", a]],
                                              time.time() - start_time)

            average_loss = total_loss / num_batches
            average_accuracy = total_accuracy / num_batches

            # Print the stats gained in the epoch
            NetworkPrint.epoch_end(epoch + 1, [["Average loss", average_loss], ["Average accuracy", average_accuracy]],
                                   time.time() - start_time)

            # Add loss and accuracy to TensorBoard

            epoch_loss_summary = tf.Summary()
            epoch_loss_summary.value.add(tag='epoch_loss', simple_value=average_loss)
            training_writer.add_summary(epoch_loss_summary, epoch + 1)

            epoch_accuracy_summary = tf.Summary()
            epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=average_accuracy)
            training_writer.add_summary(epoch_accuracy_summary, epoch + 1)
            training_writer.flush()
        # Training ends here
        # Save the model
        save_model(sess, ckpt_path, model_name)

    def evaluate(self, x_test, y_test):
        """
            This function tests the model with the passed data.

            Input:
                x_test: list, a list of input sequences to be fed in the model;
                y_test: list, a list of one-hot encoded output sequences to be fed in the model.

            Output:
                average_loss: float, the average loss (NLL) gained after evaluating that passed data.
        """

        sess = tf.Session()

        # We print that the testing has started
        NetworkPrint.testing_start()

        # We initialize the variables
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        # Adding a writer so we can visualize accuracy and loss on TensorBoard
        testing_writer = tf.summary.FileWriter(output_path + "/testing")

        # Restore the session
        restore_model(sess, ckpt_path)

        # We calculate the number of batches
        num_batches = len(x_test) // batch_size

        total_loss = 0
        total_accuracy = 0

        start_time = time.time()

        for i in range(num_batches):
            # Get a batches of data
            x_batch = get_batch(x_test, i, batch_size, fixed_batch_size)
            y_batch = get_batch(y_test, i, batch_size, fixed_batch_size)

            # Get the sequence lengths of the input sequences
            sequence_lengths = get_sequence_lengths(x_batch)

            l, a = sess.run([self.loss, self.accuracy], feed_dict={self.x: x_batch,
                                                                   self.y: y_batch,
                                                                   self.training: False,
                                                                   self.sequence_length: sequence_lengths})

            total_loss += l
            total_accuracy += a

            # Print the batch results if it's the last batch or if step printing is turned on, and this is the step
            # to print in
            if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                    or i == num_batches - 1:
                NetworkPrint.step_results(i + 1, num_batches, [["Average loss", total_loss / (i + 1)],
                                                               ["Average accuracy", total_accuracy / (i + 1)]],
                                          time.time() - start_time)

        average_loss = total_loss / num_batches
        average_accuracy = total_accuracy / num_batches

        # Print the stats gained in the evaluation phase
        NetworkPrint.evaluation_end("testing", [["Average loss", average_loss], ["Average accuracy", average_accuracy]],
                                    time.time() - start_time)

        # We add the final loss and accuracy to TensorBoard so we don't have to dig into the console logs and nohup
        # files
        testing_loss_summary = tf.Summary()
        testing_loss_summary.value.add(tag='testing_loss', simple_value=average_loss)
        testing_writer.add_summary(testing_loss_summary, 1)

        testing_accuracy_summary = tf.Summary()
        testing_accuracy_summary.value.add(tag='testing_accuracy', simple_value=average_accuracy)
        testing_writer.add_summary(testing_accuracy_summary, 1)
        testing_writer.flush()

        # We return the average accuracy (accuracy being the main metric for the sentiment analysis data sets)
        return average_accuracy


if __name__ == '__main__':  # Main function
    # Load the IMDB data set
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, max_sequence_length = load_data(vocabulary_size, max_sequence_length)

    # From which function/class we can get the model
    model_function = SentimentAnalysisModel

    if not do_hyperparameter_optimization:  # If hyperparameter optimization is off
        # Find the optimal hidden units to use without surpassing the number of parameters
        HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                 number_of_parameters=number_of_parameters,
                                                 model_function=model_function)

        # Create the model with the optimal hidden units
        MODEL = model_function(HIDDEN_UNITS)

        # Train the model
        MODEL.fit(X_TRAIN, Y_TRAIN)

        # Test the last saved model
        MODEL.evaluate(X_TEST, Y_TEST)
    else:  # If hyperparameter optimization is on
        # This probably needs better hyperparameter config (didn't update it yet, because didn't run it yet)

        # What we need to do with "hp.choice" variables
        lr_choice = [0.1, 0.05, 0.01, 0.005, 0.001]
        num_layers_choice = [1, 2, 3]
        batch_choice = [1, 2, 4, 8, 16, 32, 64]
        # We need this, so we can print the hp.choice answers normally
        choices = {
            'lr': lr_choice,
            'num_layers': num_layers_choice,
            'batch': batch_choice
        }

        # Define the space that will be passed into the hyperopt optimizing functions
        space = [
            hp.choice('lr', lr_choice),
            hp.choice('num_layers', num_layers_choice),
            hp.choice('batch', batch_choice)
        ]

        def objective(lr, num_layers, batch):
            # The function inputs must be in the same order as they are specified in the space variable
            # This function does the same steps as the above code (when hyperparameter optimization is off), but it has
            # to set the passed variables (some of them need some additional actions) and return the metric that has to
            # be minimized while doing the hyperparameter optimization

            # We'll optimize these parameters (we need to set them globally, because we use global variables in some of
            # the model functions, so the code is clearer and it doesn't need too many variables in each function)
            global learning_rate, number_of_layers, batch_size
            learning_rate = lr
            number_of_layers = num_layers
            batch_size = batch

            # We set an output path that includes the configuration, so we can later see the values in TensorBoard
            global output_path
            output_path = f"{log_path}{model_name}/{current_time}/lr{lr}layers{num_layers}batch{batch}"

            global HIDDEN_UNITS, model_function
            # Find the optimal hidden units to use without surpassing the number of parameters
            HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                     number_of_parameters=number_of_parameters,
                                                     model_function=model_function)

            # Create the model with the optimal hidden units
            model = model_function(HIDDEN_UNITS)

            # Train the model
            model.fit(X_TRAIN, Y_TRAIN)

            # Test the last saved model (it returns testing accuracy, and hyperopt needs something to minimize, so we
            # pass the negative accuracy)
            return - model.evaluate(X_TEST, Y_TEST)

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

        print_trials_information(tpe_trials, choices, metric="Accuracy", reverse_sign=True)
