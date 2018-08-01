# cnn_text_classification
A implementation of Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in Tensorflow.
## Requirements
+ Python 3 
+ Tensorflow > 1.0
+ Numpy
+ Nltk
+ tqdm
## Data Format
We need 2 files,one for training and one for validating.The format of the data is easy,each line in the file has two columns,the first is label,the second is text.They are segmented by '\t'.
data format example
```
1	for a long time the film succeeds with its dark , delicate treatment of these characters and its unerring respect for them . 
0	the film seems all but destined to pop up on a television screen in the background of a scene in a future quentin tarantino picture
0	far too clever by half , howard's film is really a series of strung-together moments , with all the spaces in between filled with fantasies , daydreams , memories and one fantastic visual trope after another . 
1	elvira fans could hardly ask for more . 
```
## Download GloVe embedding
[http://nlp.stanford.edu/data/glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)
unzip the data and move to the path
```
./input/data/embed/
```

## Process data
```
python process
```

## Training
To download the pretrained embdding
```
#download the GloVe embedding
sh download.sh
```

Print parameters:
```
python train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --train_data_file TRAIN_DATA_FILE
                        Data source for the train data.
  --valid_data_file VALID_DATA_FILE
                        Data source for the valid data.
  --save_embedding_file SAVE_EMBEDDING_FILE
                        Embeddings which contains the word from data
  --vocabulary_file VOCABULARY_FILE
                        Words in data
  --model_path MODEL_PATH
                        save model
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of word embedding(default: 300)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '2,3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularization lambda (default: 0.0)
  --max_sentence_len MAX_SENTENCE_LEN
                        The max length of sentence (default: 100)
  --input_layer_type INPUT_LAYER_TYPE
                        Type of input layer,CNN-rand,CNN-static,CNN-non-
                        static,CNN-multichannel (default: 'CNN-rand')
  --batch_size BATCH_SIZE
                        Batch Size (default: 16)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 20)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)

```


Train
use randomly initialized words
```
python train.py --input_layer_type 'CNN-rand'
```

use pre-trained vectors,but not fine-tuned during training
```
python train.py --input_layer_type 'CNN-static'
```

use pre-trained vectors,fine-tuned during training
```
python train.py --input_layer_type 'CNN-non-static'
```
use multichannel vectors
```
python train.py --input_layer_type 'CNN-multichannel'
```



