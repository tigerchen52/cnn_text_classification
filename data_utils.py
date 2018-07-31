
import os
import numpy as np
from tensorflow.python.platform import gfile
from tqdm import tqdm
import tensorflow as tf
import nltk

_PAD = "<pad>"
_SOS = "<sos>"
_UNK = "<unk>"
PAD_ID = 0
SOS_ID = 1
UNK_ID = 2
_START_VOCAB = [_PAD, _SOS, _UNK]


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        try:
            if type(x[0]) == tuple:
                x = zip(*x)
            x_batch += [x]
            y_batch += [y]
        except Exception as e:
            print(e)
            print(x)

    if len(x_batch) != 0:
        yield x_batch, y_batch

def pad_sentences(sentence, padding_word=PAD_ID, max_len=800):
    '''
    pads all sentences to the same length.
    :param setences:
    :param sent_len:
    :param padding_word: 
    :return: 
    '''
    if len(sentence) > max_len:
        sentence = sentence[:max_len]
    padding_num = max_len - len(sentence)
    new_sentence = sentence + [padding_word] * padding_num
    return new_sentence


def process_glove(vocab_list, save_path, size=4e5, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if not gfile.Exists(save_path):
        glove_path = os.path.join("./input/data/embed/glove.6B.{}d.txt".format(300))
        if random_init:
            glove = np.random.randn(len(vocab_list), 300)
        else:
            glove = np.zeros((len(vocab_list), 300))
        found = 0
        with open(glove_path, 'r', encoding='utf8') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))

def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        nmatrix of embeddings (np array)
    """
    return np.load(filename)["glove"]

def initialize_vocabulary(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path) as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def create_vocabulary(vocabulary_path, data_paths):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, encoding='utf8') as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 10000 == 0:
                        print("processing line %d" % counter)
                    tokens = nltk.word_tokenize(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with gfile.GFile(vocabulary_path, mode='w') as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + '\n')

def sentence_to_token_ids(sentence, vocabulary):
    line = sentence.split('\t')
    words = nltk.word_tokenize(line[1])
    return [line[0]] + [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocab):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        with gfile.GFile(data_path) as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def preprocess_data(
        data_paths=[],
        vocab_path='',
        embedding_path='',
        train_data_path='',
        valid_data_path=''
):
    create_vocabulary(vocabulary_path=vocab_path, data_paths=data_paths)
    vocab, rev_vocab = initialize_vocabulary(vocab_path)
    process_glove(rev_vocab, embedding_path)
    data_to_token_ids(data_path=train_data_path, target_path='./input/data/train_data.ids', vocab=vocab)
    data_to_token_ids(data_path=valid_data_path, target_path='./input/data/valid_data.ids', vocab=vocab)
    print('data is ready!!')


class text_dataset():
    def __init__(self, datafile, max_len):
        self.datafile = datafile
        self.max_len = max_len
        self.length=None

    def iter_file(self, filename):
        with open(filename, encoding='utf8') as f:
            for line in f:
                line = line.strip().split(" ")
                label = int(line[0])
                sentence = line[1:]
                sentence = list(map(lambda tok: int(tok), sentence))
                sentence = pad_sentences(sentence,max_len=self.max_len)
                yield sentence,label

    def __iter__(self):
        file_iter = self.iter_file(self.datafile)
        for text, label in file_iter:
            yield text, label

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length

if __name__ == '__main__':
    vocabulary_path = './input/data/vocabulary.txt'
    data_paths = ['./input/data/train_data.txt', './input/data/valid_data.txt']
    embedding_path = './input/data/embed/glove.6B.300d.npz'
    train_data_path = './input/data/train_data.txt'
    valid_data_path = './input/data/valid_data.txt'

    preprocess_data(
        data_paths=data_paths,
        vocab_path=vocabulary_path,
        embedding_path=embedding_path,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path
    )