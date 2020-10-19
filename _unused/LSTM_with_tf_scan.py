import numpy as np
import tensorflow as tf

import random
import argparse

from LSTM_with_tf_scan_utils import load_all_data

vocabulary_size = 5000  # 88583 for this dataset is the max! (?)
sequence_length = 500  # There are from 6 to 2493 words in our dataset! (?)  // max_words
batch_size = 64
num_epochs = 50
batches_in_epoch = 100  # Currently in each epoch it will this many times take random batch_size batches and train
state_size = 512  # This will also be our embedding size! (?)
num_classes = 2
learning_rate = 0.001
ckpt_path = 'ckpt/'
model_name = 'lstm'


class LSTM_model():

    def __init__(self):

        def __graph__():
            tf.reset_default_graph()  # Build graph

            x = tf.placeholder(shape=[None, None], dtype=tf.int32)  # Entry point for inputs
            y = tf.placeholder(shape=[None], dtype=tf.int32)  # Entry point for outputs

            # Embeddings, in the first line we could use 2nd as embedding size, but we just use another variable
            embs = tf.get_variable('emb', [vocabulary_size, state_size])  # The inputs are transformed into embeddings
            rnn_inputs = tf.nn.embedding_lookup(embs, x)  # Is the size len(x) x state_size ???

            # Initial hidden state. The shape of initial state is of form [2, batch_size, state_size].
            # Set during execution of graph. 2 because there is s and c ("value and memory")???
            init_state = tf.placeholder(shape=[2, None, state_size], dtype=tf.float32, name='initial_state')

            # Initializer
            xav_init = tf.contrib.layers.xavier_initializer

            # Parameters. 4 because there are 4 gates
            W = tf.get_variable('W', shape=[4, state_size, state_size], initializer=xav_init())  # From l step
            U = tf.get_variable('U', shape=[4, state_size, state_size], initializer=xav_init())  # For input
            b = tf.get_variable('b', shape=[2, state_size], initializer=tf.constant_initializer(0.))  # Extra for st, ct

            def step(prev, x_):  # 1 LSTM Step
                # Gather the previous internal state and output state
                st_1, ct_1 = tf.unstack(prev)

                # GATES
                # 1. Input gate
                i = tf.sigmoid(tf.matmul(x_, U[0]) + tf.matmul(st_1, W[0]))
                # 2. Forget gate
                f = tf.sigmoid(tf.matmul(x_, U[1]) + tf.matmul(st_1, W[1]))
                # 3. output gate
                o = tf.sigmoid(tf.matmul(x_, U[2]) + tf.matmul(st_1, W[2]))
                # 4. Gate weights
                g = tf.tanh(tf.matmul(x_, U[3]) + tf.matmul(st_1, W[3]))

                # New internal cell state
                ct = ct_1 * f + g * i + b[0]

                # Output state
                st = tf.tanh(ct) * o + b[1]
                return tf.stack([st, ct])

            # The dimensions of tensor rnn_inputs, are shuffled, to expose the sequence length dimension as the 0th
            # dimension, to enable iteration over the elements of the sequence.
            # The tensor of form [batch_size, seqlen, state_size], is transposed to [seqlen, batch_size, state_size]
            states = tf.scan(step,
                             tf.transpose(rnn_inputs, [1, 0, 2]),
                             initializer=init_state)

            # Predictions
            V = tf.get_variable('V', shape=[state_size, num_classes], initializer=xav_init())
            bo = tf.get_variable('bo', shape=[num_classes], initializer=tf.constant_initializer(0.))

            # Get logits
            last_state = states[-1]
            state = last_state[0]  # From st and ct picks the output (?)
            logits = tf.matmul(state, V) + bo

            # Predictions. The logits are transformed into class probabilities using the softmax function.
            predictions = tf.nn.softmax(logits)

            # Optimization
            # Following function requires logits as inputs instead of probabilities.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)  # Cross entropy losses are calculated for each time step
            loss = tf.reduce_mean(losses)  # The overall sequence loss is given by the mean of losses at each step
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  # Used to be adagrad

            # Expose symbols to class
            self.x = x
            self.y = y
            self.loss = loss
            self.train_op = train_op
            self.predictions = predictions
            self.last_state = last_state
            self.init_state = init_state

        # Build graph
        print("\nBuilding Graph...\n")
        __graph__()
        print("\n")

    def train(self, train_set):  # Function for training
        # training session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_loss = 0  # Total train loss in one epoch
            try:
                for i in range(num_epochs):
                    for j in range(batches_in_epoch):
                        xs, ys = train_set.__next__()
                        _, train_loss_ = sess.run([self.train_op, self.loss], feed_dict={  # train_op does forward and backward propagations on training data, to iteratively minimize the loss
                                self.x: xs,
                                self.y: ys,
                                self.init_state: np.zeros([2, batch_size, state_size])
                            })
                        train_loss += train_loss_
                    print('[{}] loss : {}'.format(i, train_loss/batches_in_epoch))
                    train_loss = 0
            except KeyboardInterrupt:
                print('Interrupted by user at ' + str(i))
            # Training ends here
            # Save checkpoint
            saver = tf.train.Saver()
            saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=i)

    def validate(self, valid_set, valid_set_size):
        # Start session
        with tf.Session() as sess:
            # Initialize session
            sess.run(tf.global_variables_initializer())

            # Restore session
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            saver = tf.train.Saver()
            # If there is a correct checkpoint at the path restore it
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # Will have to delete this
            # Generate operation
            state = None
            # Enter the loop
            correct = 0
            total = 0
            for i in range(valid_set_size):
                xs, ys = valid_set.__next__()
                if state:  # If it's not the first iteration
                    feed_dict = {self.x: xs,
                                 self.y: ys,
                                 self.init_state: state_}
                else:  # If this is the first iteration then we need initial state to be zeros
                    feed_dict = {self.x: xs,
                                 self.y: ys,
                                 self.init_state: np.zeros([2, 1, state_size])}  # 2 x batch size x state size

                # Forward propagation
                preds, state_ = sess.run([self.predictions, self.last_state], feed_dict=feed_dict)

                # Set flag to true
                state = True

                # Set new word
                prediction = np.random.choice(preds.shape[-1], 1, p=np.squeeze(preds))[0]
                if ys == prediction:
                    correct += 1
                total += 1
                if i % 50 == 49:
                    print(i, "th iteration, prediction - ", prediction, ", real answer - ", ys, ". Now - ", correct, "/", total)


def parse_args():  # Parse arguments
    parser = argparse.ArgumentParser(description='LSTM RNN for sentiment analysis')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--train', action='store_true', help='train model')
    group.add_argument('-v', '--validate', action='store_true', help='validate model')
    args = vars(parser.parse_args())
    return args


# A bit dirty, but for some reason after the sample_idx line it didn't allow yield x[sample_idx], y[sample_idx]
def random_batch_generator(x, y, batch_size_):
    while True:
        # Length of data, np.arange is range(length). List from the range. Sample gets batch_size samples
        sample_idx = random.sample(list(np.arange(len(x))), batch_size_)
        x_batch = []
        y_batch = []
        for i in sample_idx:
            x_batch.append(x[i])
            y_batch.append(y[i])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        yield x_batch, y_batch


if __name__ == '__main__':  # Main function
    args = parse_args()  # Parse arguments - find out train or validate

    x_train, y_train, x_test, y_test = load_all_data(vocabulary_size, sequence_length)

    model = LSTM_model()  # Create the model

    # To train or to validate
    if args['train']:
        # Each __next__ call will give batch_size random sequences
        training_set = random_batch_generator(x_train, y_train, batch_size)

        model.train(training_set)
    elif args['validate']:
        validation_set = random_batch_generator(x_test, y_test, 1)
        set_size = x_test.shape[0]
        model.validate(validation_set, set_size)
