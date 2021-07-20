# This file implements the main functions for running the MNIST task, this includes: model creation; model
# training, model testing and the main function that controls all the flow

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
from mnist_utils import load_data
# Importing some utility functions that will help us with certain tasks
from utils import find_optimal_hidden_units, print_trainable_variables, get_batch
from utils import print_trials_information, NetworkPrint

# If you have many GPUs available, you can specify which one to use here (they are indexed from 0)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# You can check if this line makes it train faster, while still training correctly
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

# Hyperparameters
# 1. Data parameters
data_set_name = "SequentialMNIST"  # string, one of these ["SequentialMNIST", "P-MNIST"]
batch_size = 64  # int, >= 1
fixed_batch_size = False  # bool
shuffle_data = True  # bool
# 2. Model parameters
cell_name = "RRU"  # string, one of these ["RRU", "GRRUA", "GRU", "LSTM", "MogrifierLSTM"]
HIDDEN_UNITS = 128  # int, >= 1 (Probably way more than 1)
number_of_parameters = 70000  # int, >= 1 (Probably way more than 1)
learning_rate = 0.001  # float, > 0
number_of_layers = 2  # int, >= 1
output_size = 64  # int, >= 1 (Probably way more than 1)
clip_gradients = True  # bool
clip_multiplier = 5.  # float
# 3. Training/testing process parameters
num_epochs = 1000000  # int, >= 1
break_epochs_no_gain = 5  # int, >= 0
do_hyperparameter_optimization = False  # bool
optimization_runs = 100  # int, >= 1
ckpt_path = 'ckpt_mnist/'  # string
log_path = 'logdir_mnist/'  # string
log_after_this_many_steps = 0  # integer, >= 0
print_after_this_many_steps = 1  # 00  # integer, >= 0
""" Hyperparameter descriptions
    Data parameters
        data_set_name
            Choose data set to test on
        batch_size
            How many data samples we feed in a single time (MNIST maximum batch_size is 10000)
        fixed_batch_size
            Can some of the batches be with size [batch_size, 2 * batch_size) (So we don't have as much left over data)
        shuffle_data
            Should we shuffle the samples?
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
            After how many steps should we print the results of training/validating/testing (0 - don't print until the
            last step)
"""

# There are 2 classes for sentiment analysis for the "MNIST" data set
num_classes = 10
# How many time steps we will unroll in RNN. MNIST when flattened has 784 sequence length
sequence_length = 784

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
relu_layers = 1
middle_layer_size_multiplier = 2
# LSTM
forget_bias = 1.0
# Mogrifier LSTM
feature_mask_rounds = 5
feature_mask_rank = 40


# Class for solving MNIST task. You can create a model, train it and test it
class MNISTModel:

    def __init__(self, hidden_units):
        """
            This function (also a constructor) creates a machine learning model for solving the MNIST task.

            Input:
                hidden_units: int, the amount of hidden units to use for the RNN cell(s).
        """

        print("\nBuilding Graph...\n")

        # Build the graph
        tf.reset_default_graph()

        # Batch size list of integer sequences
        x = tf.placeholder(tf.float32, shape=[None, sequence_length, 1], name="x")

        # One hot labels for sentiment classification
        y = tf.placeholder(tf.float32, shape=[None, num_classes], name="y")

        # Bool value that tells us, whether or not are we in training
        training = tf.placeholder(tf.bool, name='training')

        # Create the RNN cell, corresponding to the one you chose, for example, RRU, GRU, LSTM, MogrifierLSTM
        cells = []
        for _ in range(number_of_layers):
            if cell_name in ["RRU", "GRRUA"]:  # RRU
                cell = cell_fn(hidden_units,
                               training=training,
                               relu_layers=relu_layers,
                               output_size=output_size,
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

        # Extract the batch size - this allows for variable batch size
        current_batch_size = tf.shape(x)[0]

        # Create the initial state of zeros
        initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)

        # Value will have all the outputs.
        # State contains the final state.
        value, state = tf.nn.dynamic_rnn(cell,
                                         x,
                                         initial_state=initial_state,
                                         dtype=tf.float32)

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
        # Information you can get from this graph
        self.loss = loss
        self.accuracy = accuracy
        # To call the optimization step (gradient descent)
        self.optimizer = optimizer

        print_trainable_variables()

        print("\nGraph Built...\n")

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
            This function trains the model using the training data passed.

            Input:
                x_train: list, a list of input sequences to be fed in the model;
                y_train: list, a list of output sequences to be fed in the model;
                x_valid: list, a list of input sequences to be fed in the model;
                y_valid: list, a list of output sequences to be fed in the model.
        """

        with tf.Session() as sess:
            # We print that the training has started
            NetworkPrint.training_start()

            # We initialize the variables
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Adding a writer so we can visualize accuracy and loss on TensorBoard
            merged_summary = tf.summary.merge_all()
            training_writer = tf.summary.FileWriter(output_path + "/training")
            validation_writer = tf.summary.FileWriter(output_path + f"/validation")
            # Adding session graph to the writer, so we can look at it, if we want, in TensorBoard
            training_writer.add_graph(sess.graph)

            # We calculate the number of batches
            num_training_batches = len(x_train) // batch_size
            num_validation_batches = len(x_valid) // batch_size

            # Variables that help implement the early stopping if no validation loss decrease is observed
            epochs_no_gain = 0
            best_validation_accuracy = None

            for epoch in range(num_epochs):
                # Print that the epoch has started
                NetworkPrint.epoch_start(epoch + 1, num_epochs)

                # If necessary, shuffle data
                if shuffle_data:
                    x_train, y_train = shuffle(x_train, y_train)

                total_training_loss = 0
                total_training_accuracy = 0

                start_time = time.time()

                for i in range(num_training_batches):
                    # Get a batches of data
                    x_batch = get_batch(x_train, i, batch_size, fixed_batch_size)
                    y_batch = get_batch(y_train, i, batch_size, fixed_batch_size)

                    feed_dict = {
                        self.x: x_batch,
                        self.y: y_batch,
                        self.training: True
                    }

                    # If we need to log this batch, we also add summary to the sess.run
                    if log_after_this_many_steps != 0 and i % log_after_this_many_steps == 0:
                        s, _, l, a = sess.run([merged_summary, self.optimizer, self.loss, self.accuracy],
                                              feed_dict=feed_dict)

                        # Adding the summary to TensorBoard
                        training_writer.add_summary(s, i + epoch * num_training_batches)
                    else:
                        _, l, a = sess.run([self.optimizer, self.loss, self.accuracy],
                                           feed_dict=feed_dict)

                    total_training_loss += l
                    total_training_accuracy += a

                    # Print the batch results if it's the last batch or if step printing is turned on, and this is the
                    # step to print in
                    if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0) \
                            or i == num_training_batches - 1:
                        NetworkPrint.step_results(i + 1, num_training_batches, [["Loss", l], ["Accuracy", a]],
                                                  time.time() - start_time)

                average_training_loss = total_training_loss / num_training_batches
                average_training_accuracy = total_training_accuracy / num_training_batches

                # Print the stats gained in the epoch
                NetworkPrint.epoch_end(epoch + 1, [["Average loss", average_training_loss],
                                                   ["Average accuracy", average_training_accuracy]],
                                       time.time() - start_time)

                # Add loss and accuracy to TensorBoard

                epoch_loss_summary = tf.Summary()
                epoch_loss_summary.value.add(tag='training_epoch_loss', simple_value=average_training_loss)
                training_writer.add_summary(epoch_loss_summary, epoch + 1)

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
                    # Get batches of data
                    x_batch = get_batch(x_valid, i, batch_size, fixed_batch_size)
                    y_batch = get_batch(y_valid, i, batch_size, fixed_batch_size)

                    feed_dict = {
                        self.x: x_batch,
                        self.y: y_batch,
                        self.training: False
                    }

                    l, a = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)

                    total_validation_loss += l
                    total_validation_accuracy += a

                    # Print the batch results if it's the last batch or if step printing is turned on, and this is the
                    # step to print in
                    if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0) \
                            or i == num_validation_batches - 1:
                        NetworkPrint.step_results(i + 1, num_validation_batches, [["Loss", l], ["Accuracy", a]],
                                                  time.time() - start_time)

                average_validation_loss = total_validation_loss / num_validation_batches
                average_validation_accuracy = total_validation_accuracy / num_validation_batches

                # Print the stats gained in the epoch
                NetworkPrint.epoch_end(epoch + 1, [["Average loss", average_validation_loss],
                                                   ["Average accuracy", average_validation_accuracy]],
                                       time.time() - start_time)

                # Add loss and accuracy to TensorBoard

                epoch_loss_summary = tf.Summary()
                epoch_loss_summary.value.add(tag='validation_epoch_loss', simple_value=average_validation_loss)
                validation_writer.add_summary(epoch_loss_summary, epoch + 1)

                epoch_accuracy_summary = tf.Summary()
                epoch_accuracy_summary.value.add(tag='validation_epoch_accuracy',
                                                 simple_value=average_validation_accuracy)
                validation_writer.add_summary(epoch_accuracy_summary, epoch + 1)
                validation_writer.flush()

                # Training and validation for the epoch is done, check if the validation loss was better in this epoch
                # than in the last one
                if best_validation_accuracy is None or average_validation_accuracy > best_validation_accuracy:
                    print(f"&&& New best validation accuracy - before: {best_validation_accuracy};"
                          f" after: {average_validation_accuracy} - saving model...")

                    best_validation_accuracy = average_validation_accuracy

                    # Save the model
                    saver = tf.compat.v1.train.Saver()
                    saver.save(sess, ckpt_path + model_name + ".ckpt")

                    epochs_no_gain = 0
                elif break_epochs_no_gain >= 1:  # Validation accuracy was worse. Check if break_epochs_no_gain is on
                    epochs_no_gain += 1

                    print(f"&&& No validation accuracy increase for {epochs_no_gain} epochs,"
                          f" breaking at {break_epochs_no_gain} epochs.")

                    # The accuracy wasn't increasing for break_epochs_no_gain times in a row, so we need to stop the
                    # training
                    if epochs_no_gain == break_epochs_no_gain:
                        print(f"&&& Maximum epochs without validation loss decrease reached, breaking...")
                        return

    def evaluate(self, x_test, y_test):
        """
            This function tests the model with the passed data.

            Input:
                x_test: list, a list of input sequences to be fed in the model;
                y_test: list, a list of one-hot encoded output sequences to be fed in the model.

            Output:
                average_loss: float, the average loss (NLL) gained after evaluating that passed data.
        """

        with tf.Session() as sess:
            # We print that the testing has started
            NetworkPrint.testing_start()

            # We initialize the variables
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Adding a writer so we can visualize accuracy and loss on TensorBoard
            testing_writer = tf.summary.FileWriter(output_path + "/testing")

            # Restore the session
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver = tf.compat.v1.train.Saver()
            # If there is a correct checkpoint at the path restore it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # We calculate the number of batches
            num_batches = len(x_test) // batch_size

            total_loss = 0
            total_accuracy = 0

            start_time = time.time()

            for i in range(num_batches):
                # Get a batches of data
                x_batch = get_batch(x_test, i, batch_size, fixed_batch_size)
                y_batch = get_batch(y_test, i, batch_size, fixed_batch_size)

                l, a = sess.run([self.loss, self.accuracy], feed_dict={self.x: x_batch,
                                                                       self.y: y_batch,
                                                                       self.training: False})

                total_loss += l
                total_accuracy += a

                # Print the batch results if it's the last batch or if step printing is turned on, and this is the step
                # to print in
                if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0) \
                        or i == num_batches - 1:
                    NetworkPrint.step_results(i + 1, num_batches, [["Average loss", total_loss / (i + 1)],
                                                                   ["Average accuracy", total_accuracy / (i + 1)]],
                                              time.time() - start_time)

            average_loss = total_loss / num_batches
            average_accuracy = total_accuracy / num_batches

            # Print the stats gained in the evaluation phase
            NetworkPrint.evaluation_end("testing",
                                        [["Average loss", average_loss], ["Average accuracy", average_accuracy]],
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
    # Load the specific MNIST data set
    X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, X_TEST, Y_TEST = load_data("SequentialMNIST")

    # From which function/class we can get the model
    model_function = MNISTModel

    if not do_hyperparameter_optimization:  # If hyperparameter optimization is off
        # Find the optimal hidden units to use without surpassing the number of parameters
        HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                 number_of_parameters=number_of_parameters,
                                                 model_function=model_function)

        # Create the model with the optimal hidden units
        MODEL = model_function(HIDDEN_UNITS)

        # Train the model
        MODEL.fit(X_TRAIN, Y_TRAIN, X_VALID, Y_VALID)

        # Test the last saved model
        MODEL.evaluate(X_TEST, Y_TEST)
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


        # def objective(num_params, drop, lr):  # GRU
        # def objective(num_params, drop, forget, lr):  # LSTM
        # def objective(rounds, num_params, drop, rank, lr):  # Mogrifier LSTM
        def objective(num_params, drop, middle, lr):  # RRU
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
            output_path = f"{log_path}{model_name}/IMDB/{current_time}" \
                          f"/num_params{num_params}drop{drop}middle{middle}lr{lr}"  # RRU
            # f"/rounds{rounds}num_params{num_params}drop{drop}rank{rank}lr{lr}"  # Mogrifier LSTM
            # f"/num_params{num_params}drop{drop}forget{forget}lr{lr}"  # LSTM
            # f"/num_params{num_params}drop{drop}lr{lr}"  # GRU

            global HIDDEN_UNITS, model_function
            # Find the optimal hidden units to use without surpassing the number of parameters
            HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                     number_of_parameters=number_of_parameters,
                                                     model_function=model_function)

            # Create the model with the optimal hidden units
            model = model_function(HIDDEN_UNITS)

            # Train the model
            model.fit(X_TRAIN, Y_TRAIN, X_VALID, Y_VALID)

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

        print_trials_information(hyperopt_trials=tpe_trials,
                                 hyperopt_choices=choices,
                                 round_uniform=round_uniform,
                                 metric="Accuracy",
                                 reverse_sign=True)
