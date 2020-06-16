import tensorflow as tf
import numpy as np

import argparse

from sklearn.utils import shuffle

# Importing functions to load the data in the correct format, one function for tf dataset, one function for tfds dataset
from data_processor import load_data_tdfs
from data_processor import load_data_tf

from data_processor import get_sequence_lengths

# Importing the different cells we are using, RRU being the self-made one
from BasicLSTMCell import BasicLSTMCell
from GRUCell import GRUCell
from RRUCell import RRUCell

# Hyperparameters
vocabulary_size = 10000  # 88583 for tfds is the max. 24902 for tf.keras is the max
max_sequence_length = None  # 70 - 2697 words in tf.keras and 6 - 2493 words in tdfs. This will be set automatically!
batch_size = 64
num_epochs = 10
hidden_units = 128
embedding_size = 256
num_classes = 2
learning_rate = 0.001
output_keep_prob = 0.85
ckpt_path = 'ckpt/'
log_path = 'logdir/'
# model_name = 'lstm_model'
# model_name = 'gru_model'
model_name = 'rru_model'
output_path = log_path + model_name + '/rmsprop'


class LstmModel:

    def __init__(self):

        def __graph__():
            tf.reset_default_graph()  # Build the graph

            # Batch size list of integer sequences
            x = tf.placeholder(tf.int32, shape=[None, max_sequence_length], name="x")
            # One hot labels for sentiment classification
            y = tf.placeholder(tf.int32, shape=[None, num_classes], name="y")
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
            # cell = BasicLSTMCell(hidden_units)
            # cell = GRUCell(hidden_units)
            cell = RRUCell(hidden_units)

            # Extract the batch size - this allows for variable batch size
            current_batch_size = tf.shape(x)[0]

            # Create the initial state of zeros
            initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)

            # Wrap our cell in a dropout wrapper
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.85)

            # Value will have all the outputs, we will need just the last. _ contains hidden states between the steps
            value, state = tf.nn.dynamic_rnn(cell,
                                             embed_lookup,
                                             initial_state=initial_state,
                                             dtype=tf.float32,
                                             sequence_length=sequence_length)

            # Instantiate weights
            weight = tf.get_variable("weight", [hidden_units, num_classes])
            # Instantiate biases
            bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

            '''Non-variable sequence length dynamic.rnn'''
            # value = tf.transpose(value, [1, 0, 2])  # After this it's max_time, batch_size, hidden_units
            # last = value[-1]  # Extract last output. Should be batch_size hidden_units
            '''Variable sequence length'''
            last = state

            last = tf.nn.dropout(last, rate=output_drop_prob)

            prediction = tf.matmul(last, weight) + bias  # What we actually do is calculate the loss over the batch

            # predictions -        [1,1,0,0]
            # labels -             [1,0,0,1]
            # correct_prediction - [1,0,1,0]. We want accuracy over the batch, so we create a mean over it
            correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

            # Choice our model made
            choice = tf.argmax(prediction, axis=1)

            # Calculate the loss given prediction and labels
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,
                                                                             labels=y))
            tf.summary.scalar("loss", loss)

            # Declare our optimizer, we have to check which one works better.
            # Before: Adam gave better training accuracy and loss, but RMSProp gave better validation accuracy and loss
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

            # Expose symbols to class
            self.x = x
            self.y = y
            self.sequence_length = sequence_length
            self.output_drop_prob = output_drop_prob
            self.loss = loss
            self.optimizer = optimizer
            self.accuracy = accuracy
            self.prediction = prediction
            self.correct_prediction = correct_prediction
            self.choice = choice

        # Build graph
        print("\nBuilding Graph...\n")
        __graph__()
        print("\n")

    def train(self, x_train, y_train, num_batches):
        with tf.Session() as sess:
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
                for i in range(num_batches):
                    if i != num_batches - 1:
                        x_batch = x_train[i * batch_size: i * batch_size + batch_size]
                        y_batch = y_train[i * batch_size: i * batch_size + batch_size]
                    else:
                        x_batch = x_train[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                        y_batch = y_train[i * batch_size:]
                    sequence_lengths = get_sequence_lengths(x_batch)

                    s, _, l, a = sess.run([merged_summary, self.optimizer, self.loss, self.accuracy],
                                          feed_dict={self.x: x_batch,
                                                     self.y: y_batch,
                                                     self.sequence_length: sequence_lengths,
                                                     self.output_drop_prob: 1 - output_keep_prob})

                    train_writer.add_summary(s, i + epoch * num_batches)
                    total_loss += l
                    if i == 0:
                        total_accuracy = a
                    else:
                        total_accuracy = (total_accuracy * i + a) / (i + 1)

                    if i > 0 and i % 100 == 0:
                        print("STEP", i, "of", num_batches, "LOSS:", l, "ACC:", a)

                print("   Epoch", epoch + 1, ": accuracy - ", total_accuracy, ": loss - ", total_loss)
            # Training ends here
            # Save checkpoint
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=i)

    def validate(self, x_test, y_test, num_batches):
        with tf.Session() as sess:
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
                    x_batch = x_test[i * batch_size: i * batch_size + batch_size]
                    y_batch = y_test[i * batch_size: i * batch_size + batch_size]
                else:
                    x_batch = x_test[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                    y_batch = y_test[i * batch_size:]
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
            print("Final validation stats. Loss:", total_loss, "Accuracy:", total_accuracy)


def parse_args():  # Parse arguments
    parser = argparse.ArgumentParser(description='Different RNN cell comparison for IMDB review sentiment analysis')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--train', action='store_true', help='train model')
    group.add_argument('-v', '--validate', action='store_true', help='validate model')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':  # Main function
    ARGS = parse_args()  # Parse arguments - find out train or validate

    # Load or dataset
    # X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, WORD_TO_ID, ID_TO_WORD, T, max_sequence_length = load_data_tdfs(vocabulary_size)
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, max_sequence_length = load_data_tf(vocabulary_size)

    # '''
    NUM_BATCHES = len(X_TRAIN) // batch_size

    model = LstmModel()  # Create the model

    # To train or to validate
    if ARGS['train']:
        model.train(X_TRAIN, Y_TRAIN, NUM_BATCHES)
    elif ARGS['validate']:
        model.validate(X_TEST, Y_TEST, NUM_BATCHES)
    # '''
