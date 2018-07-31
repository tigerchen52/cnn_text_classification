import data_utils
import tensorflow as tf
import os

class CNN(object):
    def __init__(
            self,
            ntags=2,
            word_num=100,
            word_dim=50,
            filter_sizes='2,3,4',
            kernel_num=100,
            learning_rate_base=0.001,
            batch_size=5,
            sent_len=100,
            l2_alpha=1e-4,
            word_embedding=None,
            input_type='CNN-static',
            vocab=None,
            dropout_prob=0.5,
            epoch=20,
            model_path=''
    ):

        self.ntags = ntags
        self.word_num = word_num
        self.word_dim = word_dim
        self.filter_sizes = [int(e) for e in filter_sizes.split(',')]
        self.kernel_num = kernel_num
        self.learning_rate_base = learning_rate_base
        self.batch_size = batch_size
        self.sent_len = sent_len
        self.l2_alpha = l2_alpha
        self.pre_embedding = None
        self.input_type = input_type
        self.vocab = vocab
        self.dropout_prob = dropout_prob
        self.epoch = epoch
        self.model_path = model_path

        if self.input_type == 'CNN-rand':
            self.pre_embedding = None
            self.train_able = True
        elif self.input_type == 'CNN-static':
            self.pre_embedding = word_embedding
            self.train_able = False
        elif self.input_type == 'CNN-non-static':
            self.pre_embedding = word_embedding
            self.train_able = True
        elif self.input_type == 'CNN-multichannel':
            self.pre_embedding = word_embedding
            self.train_able = True


        self.init_graph()

    def place_holder(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.sent_len], name="input_x")

        self.input_y = tf.placeholder(tf.int64, shape=[None], name="input_y")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

    def embedding(self):
        with tf.variable_scope('word_embedding'):
            if self.pre_embedding is None:
                self._W_emb = tf.get_variable(name='embedding', shape=[self.word_num, self.word_dim],
                                              initializer=tf.random_uniform_initializer(-1.0, 1.0), trainable=self.train_able)
            else:
                self._W_emb = tf.Variable(self.pre_embedding, name='_word_embeddings', dtype=tf.float32,
                                          trainable=self.train_able)
            word_embedding = tf.nn.embedding_lookup(self._W_emb, self.input_x)

            # 3 channel
            self.word_embedding = tf.expand_dims(word_embedding, -1)
            if self.input_type == 'CNN-multichannel':
                self._W_emb_2 =  tf.Variable(self.pre_embedding, name='_word_embeddings_f', dtype=tf.float32, trainable=False)
                word_embedding2 = tf.nn.embedding_lookup(self._W_emb_2, self.input_x)
                word_embedding2 = tf.expand_dims(word_embedding2, -1)
                self.word_embedding = tf.concat([self.word_embedding, word_embedding2], axis=3)

            self.input_dim = self.word_dim

    def conv_layer(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('conv-maxpool-%s' % filter_size, reuse=tf.AUTO_REUSE):
                filter_shape = [filter_size, self.input_dim, 1, self.kernel_num]
                W = tf.get_variable(name='conv_w', shape=filter_shape,
                                    initializer=tf.truncated_normal_initializer(-0.1, 0.1))
                b = tf.get_variable(name='conv_b', shape=[self.kernel_num], initializer=tf.constant_initializer(0.1))

                conv = tf.nn.conv2d(
                    self.word_embedding,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )

                activation = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                conv_len = activation.get_shape()[1]

                pooled = tf.nn.max_pool(
                    activation,
                    ksize=[1, conv_len, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool'
                )

                pooled_outputs.append(pooled)

        self.num_filters_total = self.kernel_num * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 1)
        self.final_feature = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        with tf.name_scope("dropout"):
            self.final_feature = tf.nn.dropout(self.final_feature, self.dropout)

    def fc_layer(self):
        with tf.variable_scope("proj"):
            self.fc_w = tf.get_variable(name='fc_w', shape=[self.num_filters_total, self.ntags],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc_b = tf.get_variable(name='fc_b', shape=[self.ntags], initializer=tf.constant_initializer(0.1))

        self.fc = tf.matmul(self.final_feature, self.fc_w) + fc_b
        self.logits = self.fc

    def train_op(self):

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)

        self.l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_alpha),
            weights_list=tf.trainable_variables())
        self.loss = tf.reduce_mean(losses) + self.l2_loss

        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_base, global_step, 200, 0.99, staircase=True)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)


        self.pred = tf.nn.softmax(self.logits)
        self.pred_index = tf.argmax(self.pred, 1)
        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))


    def init_graph(self):
        tf.reset_default_graph()
        self.place_holder()
        self.embedding()
        self.conv_layer()
        self.fc_layer()
        self.train_op()

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session(self, dir_model):
        if not os.path.exists(dir_model):
            os.makedirs(dir_model)
        self.saver.save(self.sess, dir_model)

    def restore_session(self, dir_model):
        self.saver.restore(self.sess, dir_model)

    def close_session(self):
        self.sess.close()

    def train(self, train, valid):
        accuracy = 0.0
        for e in range(self.epoch):
            for i, (data, y) in enumerate(data_utils.minibatches(train, self.batch_size)):
                logits, _, lr, loss = self.sess.run([self.logits,self.train_step, self.learning_rate, self.loss], feed_dict=

                {
                    self.input_x: data,
                    self.input_y: y,
                    self.dropout: self.dropout_prob
                })
                if i % 100 == 0:
                    acc_test = self.evaluate(valid)
                    if acc_test > accuracy:
                        self.save_session(self.model_path)
                    print('This is the ' + str(e) + ' epoch training, the ' + str(i) + ' batch data，learning rate  = ' + str(round(lr, 5)) +
                          ', loss = ' + str(round(loss, 2)) + ', accuracy = ' + str(acc_test))

    def predict(self, sentences):
        pred, pred_index = self.sess.run([self.pred, self.pred_index], feed_dict=
        {

            self.input_x: sentences,
            self.dropout: 1.0
        })

        return pred_index

    def evaluate(self, valid):
        acc_total, loss_total, cnt = 0, 0, 0
        for i, (data, y) in enumerate(data_utils.minibatches(valid, self.batch_size)):
            cnt += 1
            acc = self.sess.run(self.accuracy, feed_dict={
                self.input_x: data,
                self.input_y: y,
                self.dropout: 1.0
            })
            acc_total += self.batch_size * acc
        acc_valid = round(acc_total * 1.0 / len(valid), 3)
        return acc_valid

if __name__ == '__main__':
    vocabulary_path = './input/data/vocabulary.txt'
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocabulary_path)
    embed_path = './input/data/embed/glove.6B.300d.npz'
    embeddings = data_utils.get_trimmed_glove_vectors(embed_path)
    model = CNN(
        batch_size=10,
        word_embedding=embeddings,
        sent_len=100,
        input_type='CNN-static',
        word_num=len(rev_vocab),
        word_dim=300,
        vocab=vocab
    )
    train_data = data_utils.text_dataset('./input/data/train_data.ids', 100)
    valid_data = data_utils.text_dataset('./input/data/valid_data.ids', 100)
    print('train set={a},valid set={b}'.format(a=train_data.__len__(), b=valid_data.__len__()))
    model.train(train_data, valid_data)
