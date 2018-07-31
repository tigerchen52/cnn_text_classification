
import tensorflow as tf
import data_utils
import cnn_model

#parameters

# data parameters
tf.flags.DEFINE_string("train_data_file", "./input/data/train_data.txt", "Data source for the train data.")
tf.flags.DEFINE_string("valid_data_file", "./input/data/rt-polaritydata/valid_data.txt", "Data source for the valid data.")
tf.flags.DEFINE_string("save_embedding_file", "./input/data/embed/glove.6B.300d.npz", "Embeddings which contains the word from data")
tf.flags.DEFINE_string("vocabulary_file", "./input/data/vocabulary.txt", "Words in data")
tf.flags.DEFINE_string("model_path", "./model/cnn.model", "save model")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding(default: 300)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '2,3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate (default: 0.001)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("max_sentence_len", 100, "The max length of sentence (default: 100)")
tf.flags.DEFINE_string("input_layer_type", "CNN-rand", "Type of input layer,CNN-rand,CNN-static,CNN-non-static,CNN-multichannel	 (default: 'CNN-rand')")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 16)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 20)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")



FLAGS = tf.flags.FLAGS



def preprocess():
    data_utils.preprocess_data(
        data_paths=[FLAGS.train_data_file, FLAGS.valid_data_file],
        vocab_path=FLAGS.vocabulary_file,
        embedding_path=FLAGS.save_embedding_file,
        train_data_path=FLAGS.train_data_file,
        valid_data_path=FLAGS.valid_data_file
    )

def train():
    vocab, rev_vocab = data_utils.initialize_vocabulary(FLAGS.vocabulary_file)
    embeddings = data_utils.get_trimmed_glove_vectors(FLAGS.save_embedding_file)
    model = cnn_model.CNN(
        batch_size=FLAGS.batch_size,
        word_embedding=embeddings,
        sent_len=FLAGS.max_sentence_len,
        input_type=FLAGS.input_layer_type,
        word_num=len(rev_vocab),
        word_dim=FLAGS.embedding_dim,
        vocab=vocab,
        l2_alpha=FLAGS.l2_reg_lambda,
        dropout_prob=FLAGS.dropout_keep_prob,
        kernel_num=FLAGS.num_filters,
        learning_rate_base=FLAGS.learning_rate,
        epoch=FLAGS.num_epochs,
        model_path=FLAGS.model_path

    )
    train_data = data_utils.text_dataset('./input/data/train_data.ids', FLAGS.max_sentence_len)
    valid_data = data_utils.text_dataset('./input/data/valid_data.ids', FLAGS.max_sentence_len)
    print('train data size ={a}, valid data zize ={b}'.format(a=train_data.__len__(), b=valid_data.__len__()))
    model.train(train_data, valid_data)



if __name__ == '__main__':
    train()