# -*- coding: utf-8 -*-
import argparse
import logging
import pprint

from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='gensim skip-gram with negative sampling')
parser.add_argument('--is_train', type=bool, default=False,
                    help='specify train or evaluation')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = 'vectors.model'
    if args.is_train:
        # load up unzipped corpus from http://mattmahoney.net/dc/text8.zip
        sentences = word2vec.Text8Corpus('text8')
        # train the skip-gram model
        model = word2vec.Word2Vec(sentences, sg=1, size=300, min_count=20, window=5, negative=25, workers=4)
        # save trained model
        model.save(model_name)
    else:
        # load trained model
        model = word2vec.Word2Vec.load(model_name)
        # analogy task evaluation with corpus from https://goo.gl/okpDj5
        model.accuracy('questions-words.txt')
        # execute analogy task like king - man + woman = queen
        pprint.pprint(model.most_similar(positive=['woman', 'king'], negative=['man']))


