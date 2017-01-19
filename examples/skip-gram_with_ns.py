# -*- coding: utf-8 -*-
import argparse
import sys

import numpy as np
from gensim.models import word2vec
from gensim.models.doc2vec import Word2Vec
from keras.layers import Activation, Embedding, Merge, Reshape
from keras.models import Sequential
from keras.preprocessing.sequence import skipgrams, make_sampling_table
from keras.preprocessing.text import Tokenizer, base_filter

from utils import maybe_download, unzip, read_analogies

parser = argparse.ArgumentParser(description='Keras skip-gram with negative sampling')
parser.add_argument('--save_path', type=str, default='vectors.txt',
                    help='Directory to write the model.')
parser.add_argument('--eval_data', type=str, default=None,
                    help='Analogy questions. '
                    'See README.md for how to get questions-words.txt.')
parser.add_argument('--embedding_size', type=int, default=200,
                    help='The embedding dimension size.')
parser.add_argument('--epochs_to_train', type=int, default=5,
                    help='Number of epochs to train.'
                    'Each epoch processes the training data once completely.')
parser.add_argument('--num_neg_samples', type=int, default=5,
                    help='Negative samples per training example.')
parser.add_argument('--window_size', type=int, default=4,
                    help='The number of words to predict to the left and right '
                    'of the target word.')
parser.add_argument('--min_count', type=int, default=5,
                    help='The minimum number of word occurrences for it to be '
                    'included in the vocabulary.')
parser.add_argument('--sampling_factor', type=float, default=1e-3,
                    help='Subsample threshold for word occurrence. Words that appear '
                    'with higher frequency will be randomly down-sampled. Set '
                    'to 0 to disable.')
args = parser.parse_args()


zip_filename = maybe_download('http://mattmahoney.net/dc/text8.zip')
text_file = unzip(zip_filename)
sentences = word2vec.Text8Corpus(text_file)
sentences = [' '.join(sent) for sent in sentences]
tokenizer = Tokenizer(filters=base_filter() + "'")
tokenizer.fit_on_texts(sentences)
sentences = tokenizer.texts_to_sequences(sentences)
V = len(tokenizer.word_index) + 1


def build_model():
    target_word = Sequential()
    target_word.add(Embedding(V, args.embedding_size, input_length=1))

    context = Sequential()
    context.add(Embedding(V, args.embedding_size, input_length=1))

    model = Sequential()
    model.add(Merge([target_word, context], mode='dot', dot_axes=2))
    model.add(Reshape((1,), input_shape=(1, 1)))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    return model


def train_model(model):
    sampling_table = make_sampling_table(V, sampling_factor=args.sampling_factor)
    for epoch in range(args.epochs_to_train):
        loss = 0.
        for i, sent in enumerate(sentences):
            print('{}/{}'.format(i, len(sentences)))
            couples, labels = skipgrams(sequence=sent, vocabulary_size=V, window_size=args.window_size,
                                        negative_samples=args.num_neg_samples, sampling_table=sampling_table)
            if couples:
                words, contexts = zip(*couples)
                words = np.array(words, dtype=np.int32)
                contexts = np.array(contexts, dtype=np.int32)
                y = np.array(labels, dtype=np.int32)
                loss += model.train_on_batch([words, contexts], y)
        print('num epoch: {} loss: {}'.format(epoch, loss))

    return model


def save_model(model):
    with open(args.save_path, 'w') as f:
        f.write(' '.join([str(V - 1), str(args.embedding_size)]))
        f.write('\n')
        vectors = model.get_weights()[0]
        for word, i in tokenizer.word_index.items():
            f.write(word)
            f.write(' ')
            f.write(' '.join(map(str, list(vectors[i, :]))))
            f.write('\n')


def eval_model():
    w2v = Word2Vec.load_word2vec_format(args.save_path, binary=False)
    word2id = dict([(w, i) for i, w in enumerate(w2v.index2word)])
    analogy_questions = read_analogies(args.eval_data, word2id)
    correct = 0
    total = len(analogy_questions)
    for question in analogy_questions:
        a, b, c, d = question  # E.g. [Athens, Greece, Baghdad, Iraq]
        analogies = w2v.most_similar(positive=[b, c], negative=[a], topn=4)
        for analogy in analogies:
            word, _ = analogy
            if d == word:
                # Predicted Correctly!
                correct += 1
                break
    print('Eval %4d/%d accuracy = %4.1f%%' % (correct, total, correct * 100.0 / total))



def main():
    """
    Train a word2vec model.
    """
    #if not args.train_data or not args.eval_data or not args.save_path:
    if not args.save_path:
        print('--train_data --eval_data and --save_path must be specified.')
        sys.exit(1)

    model = build_model()
    model = train_model(model)
    save_model(model)
    eval_model()

if __name__ == '__main__':
    main()
