# -*- coding: utf-8 -*-
import pprint
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer, base_filter
from keras.preprocessing.sequence import skipgrams
from keras.models import Sequential
from keras.layers import Dense
from gensim.models.doc2vec import Word2Vec


path = get_file('alice.txt', origin='http://www.gutenberg.org/cache/epub/11/pg11.txt')
sentences = [line.strip() for line in open(path) if line != '\n']

tokenizer = Tokenizer(filters=base_filter() + "'")
tokenizer.fit_on_texts(sentences)
corpus = tokenizer.texts_to_sequences(sentences)

V = len(tokenizer.word_index) + 1
dim = 200
window_size = 5


model = Sequential()
model.add(Dense(input_dim=V, output_dim=dim))
model.add(Dense(input_dim=dim, output_dim=V, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.summary()


def generate_data(corpus, window_size, V):
    for words in corpus:
        couples, labels = skipgrams(words, V, window_size, negative_samples=0, shuffle=True)
        if couples:
            X, y = zip(*couples)
            X = np_utils.to_categorical(X, V)
            y = np_utils.to_categorical(y, V)
            yield X, y

for epoch in range(10):
    loss = 0.
    for x, y in generate_data(corpus, window_size, V):
        loss += model.train_on_batch(x, y)

    print(epoch, loss)


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
