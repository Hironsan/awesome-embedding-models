# -*- coding: utf-8 -*-
import pprint
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Merge, Reshape, Activation
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer, base_filter
from keras.preprocessing.sequence import skipgrams
from gensim.models.doc2vec import Word2Vec


path = get_file('alice.txt', origin='http://www.gutenberg.org/cache/epub/11/pg11.txt')
corpus = open(path).readlines()

corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]
tokenizer = Tokenizer(filters=base_filter()+"'")
tokenizer.fit_on_texts(corpus)
V = len(tokenizer.word_index) + 1

dim = 200

inputs = Sequential()
inputs.add(Embedding(V, dim, input_length=1))

context = Sequential()
context.add(Embedding(V, dim, input_length=1))

model = Sequential()
model.add(Merge([inputs, context], mode='dot', dot_axes=2))
model.add(Reshape((1,), input_shape=(1, 1)))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')


for _ in range(10):
    loss = 0.
    for doc in tokenizer.texts_to_sequences(corpus):
        data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=5, negative_samples=5.)
        X = [np.array(x) for x in zip(*data)]
        Y = np.array(labels, dtype=np.int32)
        if X:
            loss += model.train_on_batch(X, Y)
    print(loss)

with open('vectors.txt', 'w') as f:
    f.write(' '.join([str(V-1), str(dim)]))
    f.write('\n')
    vectors = model.get_weights()[0]
    for word, i in tokenizer.word_index.items():
        f.write(word)
        f.write(' ')
        f.write(' '.join(map(str, list(vectors[i, :]))))
        f.write('\n')

w2v = Word2Vec.load_word2vec_format('./vectors.txt', binary=False)
pprint.pprint(w2v.most_similar(positive=['king']))
pprint.pprint(w2v.most_similar(positive=['place']))
pprint.pprint(w2v.most_similar(positive=['woman', 'king'], negative=['man']))
