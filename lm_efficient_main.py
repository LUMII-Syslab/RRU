import tensorflow as tf
import numpy as np
import time

from sklearn.utils import shuffle

from lm_efficient_utils import load_data
from lm_efficient_utils import get_window_indexes
from lm_efficient_utils import get_input_data_from_indexes

from BasicLSTMCell import BasicLSTMCell
from GRUCell import GRUCell
from RRUCell import RRUCell
from RRUCell import instance_norm
import os
#os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1" #jāpārbauda vai ir ātrāk un vai trenējas korekti!

# Hyperparameters
data_set_name = "enwik8"  # "enwik8", "text8", "pennchar", "penn"
vocabulary_size = None  # I will load this from a pickle file, so changing this here won't do a thing
window_size = 512
step_size = window_size // 2
batch_size = 64  # 1
num_epochs = 3
hidden_units = 512*3
embedding_size = 256
learning_rate = 0.001
output_keep_prob = 0.9
ckpt_path = 'ckpt_lm/'
log_path = 'logdir_lm/'
# model_name = 'lstm_model'
# model_name = 'gru_model'
model_name = 'rru_model'
from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
output_path = log_path + model_name + '/enwik8/'+current_time


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

            # Create LSTM/GRU/RRU Cell
            # cell = BasicLSTMCell(hidden_units)
            # cell = GRUCell(hidden_units)
            cell = RRUCell(hidden_units, dropout_rate=output_drop_prob)

            # Extract the batch size - this allows for variable batch size
            current_batch_size = tf.shape(x)[0]

            # Create the initial state of zeros
            initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)
            #initial_state+=np.asarray([1.0, -1.0]*(initial_state.get_shape().as_list()[-1]//2))*0.1
            #initial_state += 0.1

            # Wrap our cell in a dropout wrapper
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.85)

            # Value will have all the outputs. State contains hidden states between the steps.
            #embed_lookup = tf.unstack(embed_lookup, axis=1)
            value, state = tf.nn.dynamic_rnn(cell,
                                             embed_lookup,
                                             initial_state=initial_state,
                                             dtype=tf.float32)

            # Instantiate weights
            gate_img = tf.expand_dims(value[0:1, :, :], -1)
            #tf.summary.image("mem", gate_img, max_outputs=16)
            tf.summary.image("mem", tf.transpose(gate_img, [0, 2, 1, 3]), max_outputs=16)
            tf.summary.histogram("s_mul", tf.sigmoid(cell.S_bias_variable)*1.5)
            tf.summary.scalar("stateWeight", cell.prev_state_weight)
            tf.summary.scalar("W_mul", cell._W_mul)

            weight = tf.get_variable("weight", [hidden_units, vocabulary_size])
            # Instantiate biases
            bias = tf.Variable(tf.constant(0.0, shape=[vocabulary_size]))

            # [batch_size, max_time, hidden_units] -> [batch_size x max_time, hidden_units]
            #value = tf.stack(value, axis=1)
            #value -= tf.reduce_mean(value, [-1], keepdims=True)
            #value = instance_norm(value)  # If you can't get upper part working

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

            # Declare our optimizer, we have to check which one works better.
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

            # Expose symbols to class
            self.x = x
            self.y = y
            self.output_drop_prob = output_drop_prob
            self.loss = loss
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
        #tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

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

            for epoch in range(num_epochs):
                print(f"------ Epoch {epoch + 1} out of {num_epochs} ------")

                indexes = shuffle(indexes)

                total_loss = 0
                total_accuracy = 0
                total_bpc = 0

                start_time = time.time()

                for i in range(num_batches):
                    if i != num_batches - 1:
                        x_batch = indexes[i * batch_size: i * batch_size + batch_size]
                    else:
                        x_batch = indexes[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                    # Now we have batch of integers to look in text from
                    x_batch, y_batch = get_input_data_from_indexes(train_data, x_batch, window_size)

                    s, _, l, b, a = sess.run([merged_summary, self.optimizer, self.loss, self.bpc, self.accuracy],
                                             feed_dict={self.x: x_batch,
                                             self.y: y_batch,
                                             self.output_drop_prob: 1 - output_keep_prob})

                    train_writer.add_summary(s, i + epoch * num_batches)
                    total_loss += l
                    if i == 0:
                        total_accuracy = a
                        total_bpc = b
                    else:
                        total_accuracy = (total_accuracy * i + a) / (i + 1)
                        total_bpc = (total_bpc * i + b) / (i + 1)

                    if i > 0 and ((i + 1) % 100 == 0 or i == num_batches - 1):
                        print(f"Step {i + 1} of {num_batches} | Loss: {l}, BPC: {b}, Accuracy: {a}, TimeFromStart: {time.time() - start_time}")

                print(f"   Epoch {epoch + 1} | Loss: {total_loss}, BPC: {total_bpc}, Accuracy: {total_accuracy}, TimeSpent: {time.time() - start_time}")

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

                    total_loss = 0
                    total_accuracy = 0
                    total_bpc = 0

                    start_time = time.time()

                    for i in range(num_validation_batches):
                        if i != num_validation_batches - 1:
                            x_batch = validation_indexes[i * batch_size: i * batch_size + batch_size]
                        else:
                            x_batch = validation_indexes[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                        # Now we have batch of integers to look in text from
                        x_batch, y_batch = get_input_data_from_indexes(valid_data, x_batch, window_size)

                        l, b, a = sess.run([self.loss, self.bpc, self.accuracy], feed_dict={self.x: x_batch,
                                                                                            self.y: y_batch,
                                                                                            self.output_drop_prob: 0.})
                        total_loss += l
                        if i == 0:
                            total_accuracy = a
                            total_bpc = b
                        else:
                            total_accuracy = (total_accuracy * i + a) / (i + 1)
                            total_bpc = (total_bpc * i + b) / (i + 1)

                        if i > 0 and ((i + 1) % 100 == 0 or i == num_validation_batches - 1):
                            print(f"Step {i + 1} of {num_validation_batches} | Loss: {total_loss}, BPC: {total_bpc}, Accuracy: {total_accuracy}, TimeFromStart: {time.time() - start_time}")
                    print(f"Final validation stats | Loss: {total_loss}, BPC: {total_bpc}, Accuracy: {total_accuracy}, TimeSpent: {time.time() - start_time}")

                    epoch_accuracy_summary = tf.Summary()
                    epoch_accuracy_summary.value.add(tag='epoch_accuracy', simple_value=total_accuracy)
                    validation_writer.add_summary(epoch_accuracy_summary, epoch + 1)

                    epoch_bpc_summary = tf.Summary()
                    epoch_bpc_summary.value.add(tag='epoch_bpc', simple_value=total_bpc)
                    validation_writer.add_summary(epoch_bpc_summary, epoch + 1)
            # Training ends here
            # Save checkpoint
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, ckpt_path + model_name + ".ckpt", global_step=i)

    def evaluate(self, data):
        with tf.Session() as sess:
            print("|*|*|*|*|*| Starting testing... |*|*|*|*|*|")

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # I think we need to test with step_size = step_size, right? Not 1
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

            start_time = time.time()

            for i in range(num_batches):
                if i != num_batches - 1:
                    x_batch = indexes[i * batch_size: i * batch_size + batch_size]
                else:
                    x_batch = indexes[i * batch_size:]  # Run the remaining sequences (that aren't full batch_size)
                # Now we have batch of integers to look in text from
                x_batch, y_batch = get_input_data_from_indexes(data, x_batch, window_size)

                l, b, a = sess.run([self.loss, self.bpc, self.accuracy], feed_dict={self.x: x_batch,
                                                                                    self.y: y_batch,
                                                                                    self.output_drop_prob: 0.})
                total_loss += l
                if i == 0:
                    total_accuracy = a
                    total_bpc = b
                else:
                    total_accuracy = (total_accuracy * i + a) / (i + 1)
                    total_bpc = (total_bpc * i + b) / (i + 1)

                if i > 0 and ((i + 1) % 100 == 0 or i == num_batches - 1):
                    print(f"Step {i + 1} of {num_batches} | Loss: {total_loss}, BPC: {total_bpc}, Accuracy: {total_accuracy}, TimeFromStart: {time.time() - start_time}")
            print(f"Final testing stats | Loss: {total_loss}, BPC: {total_bpc}, Accuracy: {total_accuracy}, TimeSpent: {time.time() - start_time}")


if __name__ == '__main__':  # Main function
    TRAIN_DATA, VALID_DATA, TEST_DATA, vocabulary_size = load_data(data_set_name)  # Load data set

    # To see how it trains in small amounts
    '''
    TRAIN_DATA = TRAIN_DATA[:len(TRAIN_DATA) // 90]
    VALID_DATA = VALID_DATA[:len(VALID_DATA) // 5]
    TEST_DATA = TEST_DATA[:len(TEST_DATA) // 5]
    '''

    model = RNN_LM_Model()  # Create the model

    model.fit(TRAIN_DATA, VALID_DATA)  # Train the model (validating after each epoch)

    model.evaluate(TEST_DATA)  # Test the model
