import tensorflow as tf
import numpy as np
import collections
import os
import argparse
import datetime as dt

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

data_path = "c:/users/lepton/documents/programming/python/r0se/training"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()


def read_words(filename):
    with tf.io.gfile.GFile(filename, "r") as file:
        return file.read().replace("\n", "<eos>").split() # something up here??


def build_vocab(filename):
    """
    The idea of this method is to get the complete word-corpus and then give every word an ID. Words which often appear
    should get small ID's
    """
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    """
    Replace every word with it's ID (as decided in build_vocab())
    """
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "1.txt")
    valid_path = os.path.join(data_path, "1.txt")
    test_path = os.path.join(data_path, "1.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary



def batch_producer(raw_data, batch_size, num_steps): 
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype = tf.int32)
    data_len = tf.size(raw_data)
    batch_len = data_len//batch_size # integer division ? 
    data = tf.reshape(raw_data[0:batch_size * batch_len], [batch_size, batch_len])

    epoch_size = (batch_len-1)//num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle = False).dequeue()
    x = data[:, i*num_steps:(i+1)*num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i*num_steps + 1:(i+1)*num_steps]
    y.set_shape([batch_size, num_steps])
    return x, y 


class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data)//batch_size)-1)//num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)
    
class Model(object):
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers, dropout=0.5, init_scale=0.05):
        self.is_training=is_training
        self.input_obj = input
        self.batch_size = input.batch_size 
        self.num_steps = input.num_steps

        #create word embeddings
        with tf.device('/cpu:0'): 
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        if is_training and dropout<1:
            inputs = tf.nn.dropout(inputs, dropout)

        #create LSTM model, set up storage/extraction
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])
        
        # set up state data variable to be fed into tensorflow LSTM structure
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                                for idx in range(num_layers))

        # create LSTM cell
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias = 1.0)
        # if training add dropout warning
        if is_training and dropout < 1: 
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = dropout)
        
        # if we have layers
        if num_layers > 1: 
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple = True)
        
        # create dynamic RNN object
        output, self.state = tf.rnn.dynamic_rnn(cell, inputs, dtype = tf.float32, initial_state = rnn_tuple_state)

        # Create softmax, loss, and optimizer functions
        # reshape 
        output = tf.reshape(output, [-1, hidden_size])
        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b) # output of tensor multiplication

        # set up loss/cost function - use TensorFlow sequence to sequence loss function
        # to calculate weighted cross entrophy loss across a sequence of values 
        # reshape logits to be a 3D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # average loss over batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits, 
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps], dtype = tf.float32),
            average_across_timesteps = False, 
            average_across_batch = True
        )
        self.cost = tf.reduce_sum(loss)

        # get prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis = 1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
        self.acurracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training: 
            return 
        self.learning_rate = tf.Variable(0.0, trainable = False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step = tf.contrib.framework.get_or_create_global_step()
        )

        self.new_lr = tf.placeholder(tf.float32, shape = [])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    # Training loops
    def train(train_data, vocabulary, num_layers, num_epoch, batch_size, model_save_name, 
            learning_rate = 1.0, max_lr_epoch = 10, lr_decay = 0.93):
        # set up data and models
        training_input = Input(batch_size = batch_size, num_steps = 35, data = train_data)
        m = Model(training_input, is_training = True, hidden_size = 650, vocab_size = vocabulary, num_layers = num_layers)
        init_op = tf.global_variables_initializer()

        # Create input and model object 
        orig_decay = lr_decay
        # keep after creation of training_input or program will hang
        with tf.Session() as sess:
            # start threads
            sess.run([init_op])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            saver = tf.train.saver
        
            # enter epoch training loop 
            for epoch in range(num_epoch):
                new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
                m.assign_lr(sess, learning_rate * new_lr_decay)
                current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
                for step in range(training_input.epoch_size):
                    if step % 50 != 0 : 
                        cost, _, current_state = sess.run([m.cost, m.train_op, m.state], 
                        feed_dict = {m.init_state: current_state})
                    else: 
                        cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                        feed_dict = {m.init_state: current_state})
                        print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}".format(epoch, step, cost, acc))
                    # save a model checkpoint
                    saver.save(sess, data_path + '\\' + model_save_name, global_step = epoch)
                # do final save 
                saver.save(sess, data_path + '\\' + model_save_name + '-final')
                # close threads
                coord.request_stop()
                coord.join(threads)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

load_data()