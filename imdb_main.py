import tensorflow as tf
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
# Importing the necessary stuff for hyperparameter optimization
from hyperopt import hp, tpe, Trials, fmin

# Importing fancier optimizer(s)
# from RAdam import RAdamOptimizer
from adam_decay import AdamOptimizer_decay

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
max_sequence_length = 500  # [70-2697] words in tf.keras (used to be [6-2493] words in tdfs). None means max
batch_size = 64  # Max batch_size = 25000
num_epochs = 10
HIDDEN_UNITS = 128
number_of_parameters = 2000000  # 2 million
number_of_layers = 2
embedding_size = 32
num_classes = 2
learning_rate = 0.001
shuffle_data = True
fixed_batch_size = False
do_hyperparameter_optimization = False

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

        # Cast our label to float32. Later it will be better when it does some math (?)
        y = tf.cast(y, tf.float32)

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

        # Extract the batch size - this allows for variable batch size
        current_batch_size = tf.shape(x)[0]

        # Create the initial state of zeros
        initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)

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
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction, label_smoothing=0.1)

        # Printing trainable variables which have "kernel" in their name
        decay_vars = [v for v in tf.trainable_variables() if 'kernel' in v.name]
        for c in decay_vars:
            print(c)

        # Declare our optimizer, we have to check which one works better.
        # Before: Adam gave better training accuracy and loss, but RMSProp gave better validation accuracy and loss
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        optimizer = AdamOptimizer_decay(learning_rate=learning_rate,
                                        L2_decay=0.01,
                                        decay_vars=decay_vars).minimize(loss)
        # optimizer = RAdamOptimizer(learning_rate=learning_rate, L2_decay=0.0, epsilon=1e-8).minimize(loss)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

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
            training_writer = tf.summary.FileWriter(output_path + "/training")
            training_writer.add_graph(sess.graph)

            for epoch in range(num_epochs):
                print(f"------ Epoch {epoch + 1} out of {num_epochs} ------")
                if epoch > 0:  # If it's not the first epoch, we shuffle the data
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

                        training_writer.add_summary(s, i + epoch * num_batches)
                    else:
                        _, l, a = sess.run([self.optimizer, self.loss, self.accuracy],
                                           feed_dict=feed_dict)

                    total_loss += l
                    total_accuracy += a

                    if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                            or i == num_batches - 1:
                        print(f"Step {i + 1} of {num_batches} | "
                              f"Loss: {l}, "
                              f"Accuracy: {a}, "
                              f"Time from start: {time.time() - start_time}")

                average_loss = total_loss / num_batches
                average_accuracy = total_accuracy / num_batches
                print(f"   Epoch {epoch + 1} | "
                      f"Average loss: {average_loss}, "
                      f"Average accuracy: {average_accuracy}, "
                      f"Time spent: {time.time() - start_time}")

                epoch_loss_summary = tf.Summary()
                epoch_loss_summary.value.add(tag='epoch_loss', simple_value=average_loss)
                training_writer.add_summary(epoch_loss_summary, epoch + 1)

                epoch_accuracy_summary = tf.Summary()
                epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=average_accuracy)
                training_writer.add_summary(epoch_accuracy_summary, epoch + 1)
                training_writer.flush()
            # Training ends here
            # Save checkpoint
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, ckpt_path + model_name + ".ckpt")

    def evaluate(self, x_test, y_test):
        with tf.Session() as sess:
            print("|*|*|*|*|*| Starting testing... |*|*|*|*|*|")

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Adding a writer so we can visualize accuracy and loss on TensorBoard
            testing_writer = tf.summary.FileWriter(output_path + "/testing")
            testing_writer.add_graph(sess.graph)

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
                total_accuracy += a

                if (print_after_this_many_steps != 0 and (i + 1) % print_after_this_many_steps == 0)\
                        or i == num_batches - 1:
                    print(f"Step {i + 1} of {num_batches} | "
                          f"Average loss: {total_loss / (i + 1)}, "
                          f"Average accuracy: {total_accuracy / (i + 1)}, "
                          f"Time from start: {time.time() - start_time}")

            average_loss = total_loss / num_batches
            average_accuracy = total_accuracy / num_batches
            print(f"Final testing stats | "
                  f"Average loss: {average_loss}, "
                  f"Average accuracy: {average_accuracy}, "
                  f"Time spent: {time.time() - start_time}")

            # We add this to TensorBoard so we don't have to dig in console logs and nohups
            testing_loss_summary = tf.Summary()
            testing_loss_summary.value.add(tag='testing_loss', simple_value=average_loss)
            testing_writer.add_summary(testing_loss_summary, 1)

            testing_accuracy_summary = tf.Summary()
            testing_accuracy_summary.value.add(tag='testing_accuracy', simple_value=average_accuracy)
            testing_writer.add_summary(testing_accuracy_summary, 1)
            testing_writer.flush()

            return average_accuracy


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

    if not do_hyperparameter_optimization:
        HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                 number_of_parameters=number_of_parameters,
                                                 model_function=model_function)

        MODEL = model_function(HIDDEN_UNITS)  # Create the model

        MODEL.fit(X_TRAIN, Y_TRAIN)

        MODEL.evaluate(X_TEST, Y_TEST)
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
            output_path = f"{log_path}{model_name}/{current_time}/lr{lr}layers{num_layers}batch{batch}"

            global HIDDEN_UNITS
            global model_function

            HIDDEN_UNITS = find_optimal_hidden_units(hidden_units=HIDDEN_UNITS,
                                                     number_of_parameters=number_of_parameters,
                                                     model_function=model_function)

            model = model_function(HIDDEN_UNITS)  # Create the model

            model.fit(X_TRAIN, Y_TRAIN)  # Train the model (validating after each epoch)

            # Test the last saved model (it returns testing accuracy, and hyperopt needs something to minimize, so we
            # pass negative accuracy)
            return - model.evaluate(X_TEST, Y_TEST)

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

        print_trials_information(tpe_trials, choices, metric="Accuracy", reverse_sign=True)
