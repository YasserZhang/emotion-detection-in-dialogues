import tensorflow as tf
'''
testing...

num_seq = 2
sequence_length = 6
num_classes = 2
vocab_size = 1000
embedding_size = 50
filter_sizes = [1,2,3]
num_filters = 5
# Number of Epochs
num_epochs = 2
# Batch Size
batch_size = 2
# RNN Size
rnn_size = 20
dummy_x = np.array([[[1,2,3,4,5,6],[2,3,45,6,10,34]], [[1,2,3,4,5,6],[2,3,45,6,10,34]]])
dummy_y = np.array([[0,1],[1,0]])
drop_prob = 0.5
'''

def vocab_lookup(input_x, vocab_size, embedding_size, trainable = True):
    #input_x: [batch_size, seq_length]
    '''
    output:
        Vocab_Matrix: [vocab_size, embedding_size]
        embedded_chars_expanded: [batch_size, seq_length, embedding_size, 1]
    '''
    with tf.device('/cpu:0'), tf.variable_scope('embedding', reuse = tf.AUTO_REUSE):
        Vocab_Matrix = tf.get_variable('Vocab_M', shape=[vocab_size, embedding_size], trainable = trainable)
        embedded_words = tf.nn.embedding_lookup(Vocab_Matrix, input_x)
        embedded_words_expanded = tf.expand_dims(embedded_words, -1)
    return Vocab_Matrix, embedded_words_expanded

def conv_layers(embedded_words_expanded,sequence_length, filter_sizes, embedding_size, num_filters):
    expanded_shape = embedded_words_expanded.get_shape()
    #print(expanded_shape)
    embedded_words_expanded_flatten = tf.reshape(embedded_words_expanded, shape = [-1, *expanded_shape[2:]])
    #print(embedded_words_expanded_flatten.get_shape())
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope('conv-maxpool-%s' % filter_size), tf.variable_scope("conv_maxpool-variables-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.get_variable("W_%d" % i, shape = filter_shape, initializer = tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable("b_%d" % i, initializer = tf.constant(0.1, shape=[num_filters]))
            
            conv = tf.nn.conv2d(embedded_words_expanded_flatten,
                               W,
                               strides = [1,1,1,1],
                               padding='VALID',
                               name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            #print("h dim:", h.get_shape())
            pooled = tf.nn.max_pool(h,
                                   ksize=[1,sequence_length - filter_size + 1, 1, 1],
                                   strides =[1,1,1,1],
                                   padding='VALID',
                                   name='pool')
            pooled_outputs.append(pooled)
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    h_pool_flat = tf.reshape(h_pool_flat, [-1, expanded_shape[1], num_filters_total])
    return h_pool_flat

def get_init_cell(batch_size, rnn_size, keep_prob=1.0, num_layers = 1):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    number_of_layers = num_layers
    def lstm_cell():
        cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, reuse = tf.AUTO_REUSE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return cell

    with tf.name_scope('initialize_rnn_layers'):
        cell = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell() for _ in range(number_of_layers)])
        initial_state = cell.zero_state(batch_size, tf.float32)
        initial_state = tf.identity(initial_state, name = 'initial_state')
    return cell, initial_state

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    with tf.name_scope("rnn_output"):
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)

        final_state = tf.identity(final_state, name = 'final_state')
        #print(outputs)
        #print("final state shape:",final_state)
    return outputs, final_state

class CRNN(object):
    def __init__(self, \
                 num_seq, \
                 sequence_length, \
                 num_classes, \
                 vocab_size, \
                 embedding_size, \
                 filter_sizes, \
                 num_filters,
                 rnn_size,
                 is_train=True):
        self.input_x = tf.placeholder(tf.int32, [None, num_seq, sequence_length], name= 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.Vocab_Matrix, self.embedded_words_expanded = vocab_lookup(self.input_x, vocab_size, embedding_size, trainable = is_train)
        with tf.variable_scope("cnn", reuse = tf.AUTO_REUSE):
            self.h_pool_flat = conv_layers(self.embedded_words_expanded,sequence_length, filter_sizes, embedding_size, num_filters)

        with tf.variable_scope("rnn", reuse = tf.AUTO_REUSE):
            self.h_pool_flat = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            input_data_shape = tf.shape(self.h_pool_flat) #shape or get_shape?
            #print("h_pool_flat shape")
            #print(input_data_shape)
            cell, initial_state = get_init_cell(input_data_shape[0], rnn_size, keep_prob=self.dropout_keep_prob)
            output, self.final_state = build_rnn(cell, self.h_pool_flat)
            #output = tf.reshape(output, shape = [1, -1])
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            self.output = tf.contrib.layers.flatten(output)
            self.logits = tf.layers.dense(self.output, num_classes, activation = None, use_bias = True)
        with tf.variable_scope('loss', reuse = tf.AUTO_REUSE):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
        with tf.variable_scope("accuracy", reuse=tf.AUTO_REUSE):
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
