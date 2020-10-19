# This is an early version of lm_main and lm_utils, that had very large memory usage
import tensorflow as tf
import numpy as np

from sklearn.utils import shuffle

from _unused.lm_heavy_utils import load_data
from _unused.lm_heavy_utils import get_sequence_lengths

from cells.RRUCell import RRUCell

# Hyperparameters
data_set_name = "enwik8"  # "enwik8", "text8", "pennchar", "penn"
# I will load the bottom three from pickle files. Changing these here won't do a thing
vocabulary_size = None  # 207 is the max for enwik8. 29 is the max for text8. Both numbers include 2 for unk and pad.
window_size = None  # I used 10 for all, atm
step_size = None
batch_size = 64
num_epochs = 10
hidden_units = 1024
embedding_size = 256
learning_rate = 0.001
output_keep_prob = 0.9  # 0.85
ckpt_path = 'ckpt_lm/'
log_path = 'logdir_lm/'
# model_name = 'lstm_model'
# model_name = 'gru_model'
model_name = 'rru_model'
output_path = log_path + model_name + '/enwik8'


class RNN_LM_Model:

    def __init__(self):

        def __graph__():
            tf.reset_default_graph()  # Build the graph

            # Batch size list of integer sequences
            x = tf.placeholder(tf.int32, shape=[None, window_size], name="x")
            # One-hot labels for word prediction
            y = tf.placeholder(tf.int32, shape=[None, vocabulary_size], name="y")
            # Batch size list of sequence lengths, so we can get variable sequence length rnn
            sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
            # Output drop probability so we can pass different values depending on training/ testing
            output_drop_prob = tf.placeholder(tf.float32, name='output_drop_prob')

            # Cast our label to float32. Later it will be better when it does some math (?)
            y = tf.cast(y, tf.float32)

            # Instantiate our embedding matrix
            embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
                                    name="word_embedding")

            # Lookup embeddings
            embed_lookup = tf.nn.embedding_lookup(embedding, x)

            # Create LSTM/GRU/RRU Cell
            # cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units, state_is_tuple=True)
            # cell = tf.nn.rnn_cell.GRUCell(hidden_units)
            cell = RRUCell(hidden_units, dropout_rate=output_drop_prob)

            # Extract the batch size - this allows for variable batch size
            current_batch_size = tf.shape(x)[0]

            # Create the initial state of zeros
            initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)

            # Wrap our cell in a dropout wrapper
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.85)

            # Value will have all the outputs, we will need just the last. _ contains hidden states between the steps
            # value, (_, state) = tf.nn.dynamic_rnn(cell,  # BasicLSTMCell needs this
            value, state = tf.nn.dynamic_rnn(cell,
                                             embed_lookup,
                                             initial_state=initial_state,
                                             dtype=tf.float32,
                                             sequence_length=sequence_length)

            # Instantiate weights
            weight = tf.get_variable("weight", [hidden_units, vocabulary_size])
            # Instantiate biases
            bias = tf.Variable(tf.constant(0.0, shape=[vocabulary_size]))

            '''Non-variable sequence length dynamic.rnn'''
            # value = tf.transpose(value, [1, 0, 2])  # After this it's max_time, batch_size, hidden_units
            # last = value[-1]  # Extract last output. Should be batch_size hidden_units
            '''Variable sequence length'''
            last = state

            # last = tf.nn.dropout(last, rate=output_drop_prob)

            prediction = tf.matmul(last, weight) + bias  # What we actually do is calculate the loss over the batch

            correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

            # Calculate the loss given prediction and labels

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,
                                                                             labels=y))

            tf.summary.scalar("loss", loss)

            perplexity = tf.exp(loss)

            tf.summary.scalar("perplexity", perplexity)

            # Declare our optimizer, we have to check which one works better.
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

            # Expose symbols to class
            self.x = x
            self.y = y
            self.sequence_length = sequence_length
            self.output_drop_prob = output_drop_prob
            self.loss = loss
            self.perplexity = perplexity
            self.optimizer = optimizer
            self.accuracy = accuracy
            self.prediction = prediction
            self.correct_prediction = correct_prediction

        # Build graph
        print("\nBuilding Graph...\n")
        __graph__()
        print("\nGraph Built...\n")

    def fit(self, x_train, y_train, num_batches_train, x_valid, y_valid, num_batches_valid):
        with tf.Session() as sess:
            ''' ### TRAIN ### '''
            print("Starting training...")
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Adding a writer so we can visualize accuracy and loss on TensorBoard
            merged_summary = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(output_path)
            train_writer.add_graph(sess.graph)

            for epoch in range(num_epochs):
                print("------ Epoch", epoch + 1, "out of", num_epochs, "------")
                if epoch > 0:
                    data = list(zip(x_train, y_train))
                    shuffle(data)
                    x_train, y_train = zip(*data)

                total_loss = 0
                total_accuracy = 0
                for i in range(num_batches_train):
                    if i != num_batches_train - 1:
                        x_batch = x_train[i * batch_size: i * batch_size + batch_size]
                        y_batch = y_train[i * batch_size: i * batch_size + batch_size]
                    else:
                        x_batch = x_train[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                        y_batch = y_train[i * batch_size:]
                    sequence_lengths = get_sequence_lengths(x_batch)

                    s, _, l, p, a = sess.run([merged_summary, self.optimizer, self.loss, self.perplexity, self.accuracy],
                                             feed_dict={self.x: x_batch,
                                             self.y: y_batch,
                                             self.sequence_length: sequence_lengths,
                                             self.output_drop_prob: 1 - output_keep_prob})

                    train_writer.add_summary(s, i + epoch * num_batches_train)
                    total_loss += l
                    if i == 0:
                        total_accuracy = a
                    else:
                        total_accuracy = (total_accuracy * i + a) / (i + 1)

                    if i > 0 and i % 100 == 0:
                        print("STEP", i, "of", num_batches_train, "LOSS:", l, "PERPLEXITY:", p, "ACC:", a)

                print("   Epoch", epoch + 1, ": accuracy - ", total_accuracy, ": loss - ", total_loss)

                ''' ### VALIDATE ### '''
                self.test_or_validate(x_valid, y_valid, num_batches_valid)
            # Training ends here
            # Save checkpoint
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=i)

    def test_or_validate(self, x, y, num_batches, testing=False):
        with tf.Session() as sess:
            if testing:
                print("Starting testing...")
            else:
                print("Starting validation...")

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # Restore session
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver = tf.compat.v1.train.Saver()
            # If there is a correct checkpoint at the path restore it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            total_loss = 0
            total_accuracy = 0
            for i in range(num_batches):
                if i != num_batches - 1:
                    x_batch = x[i * batch_size: i * batch_size + batch_size]
                    y_batch = y[i * batch_size: i * batch_size + batch_size]
                else:
                    x_batch = x[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                    y_batch = y[i * batch_size:]
                sequence_lengths = get_sequence_lengths(x_batch)

                l, a = sess.run([self.loss, self.accuracy], feed_dict={self.x: x_batch,
                                                                       self.y: y_batch,
                                                                       self.sequence_length: sequence_lengths,
                                                                       self.output_drop_prob: 0.})
                total_loss += l
                if i == 0:
                    total_accuracy = a
                else:
                    total_accuracy = (total_accuracy * i + a) / (i + 1)
                if i > 0 and i % 100 == 0:
                    print("Step", i, "of", num_batches, "Loss:", total_loss, "Accuracy:", total_accuracy)
            if testing:
                print("Final testing stats. Loss:", total_loss, "Accuracy:", total_accuracy)
            else:
                print("Final validation stats. Loss:", total_loss, "Accuracy:", total_accuracy)


def one_hot_encode(labels, vocab_size):
    n_labels = len(labels)
    n_unique_labels = vocab_size  # len(np.unique(labels))
    one_hot_encoded = np.zeros((n_labels, n_unique_labels))  # Now it's [0 0] [0 0]
    one_hot_encoded[np.arange(n_labels), labels] = 1  # Now it's [1 0] [0 1]
    return one_hot_encoded


if __name__ == '__main__':  # Main function
    # Load data set
    print("Started loading data...")
    X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, X_TEST, Y_TEST, vocabulary_size, window_size, step_size = load_data("enwik8")

    # Do we have to? Or is there a way not to do this?
    print("One-hot encoding the data")
    Y_TRAIN = one_hot_encode(Y_TRAIN, vocabulary_size)
    Y_VALID = one_hot_encode(Y_VALID, vocabulary_size)
    Y_TEST = one_hot_encode(Y_TEST, vocabulary_size)

    # Getting amount of full batches
    NUM_BATCHES_TRAIN = len(X_TRAIN) // batch_size
    NUM_BATCHES_VALID = len(X_VALID) // batch_size
    NUM_BATCHES_TEST = len(X_TEST) // batch_size

    model = RNN_LM_Model()  # Create the model

    model.fit(X_TRAIN, Y_TRAIN, NUM_BATCHES_TRAIN, X_VALID, Y_VALID, NUM_BATCHES_VALID)

    model.test_or_validate(X_TEST, Y_TEST, NUM_BATCHES_TEST, testing=True)
