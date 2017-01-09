# -*- coding: utf-8 -*-
import logging

from gensim.models import word2vec

from .utils import maybe_download


# for logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load up unzipped corpus from http://mattmahoney.net/dc/text8.zip
sentences = word2vec.Text8Corpus('/tmp/text8')

# train the skip-gram model; default window=5
model = word2vec.Word2Vec(sentences, sg=1, size=200, min_count=20, window=5, negative=5)

# ... and some hours later... just as advertised...
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)