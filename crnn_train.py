import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import time
import os
import datetime
import pickle
from data_helper.helper import load_sub_dialogues, batch_iter, tokenizer
from cnn_rnn.cnn_rnn import CRNN

tf.reset_default_graph()

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
#tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_sequences", 5, "the size of a sequence of utterances extracted from a dialogue")
# Training parameters

tf.flags.DEFINE_integer("batch_size", 30, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 7, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 200, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("rnn_size", 512, "Number of units in LSTM")
#tf.flags.DEFINE_boolean("is_training", True, "the dataset is used for training")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

with open('vocabs.pickle', 'rb') as f:
    vocabs = pickle.load(f)
#load in train and dev data
train_x, train_y = load_sub_dialogues('./data/new_train.json')
dev_x, dev_y = load_sub_dialogues('./data/new_dev.json')
x_text = train_x + dev_x
y = train_y + dev_y
y = np.copy(np.array(y))
#transform y
labels = np.unique(y)
y = np.array([list(map(int, labels == y_)) for y_ in y])


# Build vocabulary
max_document_length = max([len(x.split()) for dialogue in x_text for x in dialogue])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, tokenizer_fn = tokenizer)
#vocab_processor.fit([text for x in x_text for text in x])
vocab_processor.fit(vocabs)
x = np.array([list(vocab_processor.transform(text)) for text in x_text])

#dev_sample_index = -1 * int(0.1 * float(len(y)))
dev_sample_index = -len(dev_x)
x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

#test data
test_x, test_y = load_sub_dialogues('./data/new_test.json')
x_test = np.array([list(vocab_processor.transform(text)) for text in test_x])
y_test = np.array([list(map(int, labels == y_)) for y_ in test_y])
del x, y, train_x,train_y, dev_x, dev_y, test_x, test_y

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print("Test size: {:d}".format(len(y_test)))

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    #writer = tf.summary.FileWriter('graphs/cnn-rnn-train', sess.graph)
    crnn = CRNN(
            num_seq=FLAGS.num_sequences, \
            sequence_length=x_train.shape[2], \
            num_classes = y_train.shape[1], \
            vocab_size = len(vocab_processor.vocabulary_), \
            embedding_size = FLAGS.embedding_dim, \
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))), \
            num_filters = FLAGS.num_filters,\
            rnn_size = FLAGS.rnn_size,\
            is_train=False)
    '''
    crnn_dev = CRNN(
            num_seq=FLAGS.num_sequences, \
            sequence_length = x_train.shape[2], \
            num_classes = y_train.shape[1], \
            vocab_size = len(vocab_processor.vocabulary_), \
            embedding_size = FLAGS.embedding_dim, \
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))), \
            num_filters = FLAGS.num_filters, \
            rnn_size = FLAGS.rnn_size, \
            is_train=False)
    '''
    with tf.name_scope("train"):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(crnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
    
    
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    
    #Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "graphs", timestamp))
    print("Writing to {}\n".format(out_dir))
    
    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", crnn.loss)
    acc_summary = tf.summary.scalar("accuracy", crnn.accuracy)
    
    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    
    #Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
    
    sess.run(tf.global_variables_initializer())
    
    initW = np.random.normal(0,1, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
    for word in vocabs:
        idx = vocab_processor.vocabulary_.get(word)
        initW[idx] = vocabs[word]
        
    sess.run(crnn.Vocab_Matrix.assign(initW))
    
    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {crnn.input_x:x_batch,
                    crnn.input_y: y_batch,
                    crnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
        _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, crnn.loss, crnn.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        if step % 10 == 0:
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)
    
    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluate model on a dev set
        """
        feed_dict = {
            crnn.input_x: x_batch,
            crnn.input_y: y_batch,
            crnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run([global_step, train_summary_op, crnn.loss, crnn.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)
     # Generate batches
    batches = batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_step(x_dev, y_dev, writer=dev_summary_writer)
            print("")
    
    # Testing
    def test(x_batch, y_batch):
        feed_dict = {
            crnn.input_x: x_batch,
            crnn.input_y: y_batch,
            crnn.dropout_keep_prob: 1.0
        }
        predictions, loss, accuracy = sess.run(
            [crnn.predictions,crnn.loss,crnn.accuracy],
            feed_dict
        )
        return predictions, loss, accuracy
    predictions, loss, accuracy = test(x_test, y_test)
    print("\nTest Accuracy: {:g}; Loss: {:g}".format(accuracy, loss))
    with open('./predictions/crnn_predictions.pickle', 'wb') as f:
        pickle.dump(predictions, f)

# accuracy: 0.546718; loss: 1.33084