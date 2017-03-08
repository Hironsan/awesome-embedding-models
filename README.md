# awesome-embedding-models[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of awesome embedding models tutorials, projects and communities.
Please feel free to pull requests to add links.

## Table of Contents


* **[Papers](#papers)**
* **[Researchers](#researchers)**
* **[Courses and Lectures](#courses-and-lectures)**
* **[Datasets](#datasets)**
* **[Articles](#articles)**
* **[Implementations and Tools](#implementations-and-tools)**


## Papers
### Word Embeddings

**Word2vec**

* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
* [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
* [word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738)
* [word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method](https://arxiv.org/abs/1402.3722)

**GloVe**

* [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf)
* [Improving Word Representations via Global Context and Multiple Word Prototypes](http://www.aclweb.org/anthology/P12-1092)

**FastText**

* [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606v1.pdf)
* [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)

**Embedding Enhancement**

* [Retrofitting Word Vectors to Semantic Lexicons](https://arxiv.org/abs/1411.4166)
* [Better Word Representations with Recursive Neural Networks for Morphology](http://nlp.stanford.edu/~lmthang/data/papers/conll13_morpho.pdf)
* [Dependency-Based Word Embeddings](https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf)
* [Not All Neural Embeddings are Born Equal](https://arxiv.org/abs/1410.0718)
* [Two/Too Simple Adaptations of Word2Vec for Syntax Problems](http://www.cs.cmu.edu/~lingwang/papers/naacl2015.pdf)


**Comparing count-based vs predict-based method**

* [Linguistic Regularities in Sparse and Explicit Word Representations](https://www.cs.bgu.ac.il/~yoavg/publications/conll2014analogies.pdf)
* [Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors](http://www.aclweb.org/anthology/P14-1023)
* [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)


**Evaluation**

* [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)
* [Intrinsic Evaluation of Word Vectors Fails to Predict Extrinsic Performance](https://www.aclweb.org/anthology/W/W16/W16-2501.pdf)
* [Problems With Evaluation of Word Embeddings Using Word Similarity Tasks](https://arxiv.org/abs/1605.02276)
* [Improving Reliability of Word Similarity Evaluation by Redesigning Annotation Task and Performance Measure](https://arxiv.org/abs/1611.03641)
* [Evaluating Word Embeddings Using a Representative Suite of Practical Tasks](https://cs.stanford.edu/~angeli/papers/2016-acl-veceval.pdf)

### Phrase, Sentence and Document Embeddings

**Sentence**

* [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726)

**Document**

* [Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053)

### Sense Embeddings

* [SENSEMBED: Learning Sense Embeddings for Word and Relational Similarity](http://wwwusers.di.uniroma1.it/~navigli/pubs/ACL_2015_Iacobaccietal.pdf)
* [Multi-Prototype Vector-Space Models of Word Meaning](http://www.cs.utexas.edu/~ml/papers/reisinger.naacl-2010.pdf)

### Neural Language Models

* [Recurrent neural network based language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
* [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
* [Linguistic Regularities in Continuous Space Word Representations](http://www.aclweb.org/anthology/N13-1090)

## Researchers

* [Tomas Mikolov](https://scholar.google.co.jp/citations?user=oBu8kMMAAAAJ&hl=en)
* [Yoshua Bengio](https://scholar.google.co.jp/citations?user=kukA0LcAAAAJ&hl=en)
* [Yoav Goldberg](https://scholar.google.co.jp/citations?user=0rskDKgAAAAJ&hl=en)
* [Omer Levy](https://scholar.google.co.jp/citations?user=PZVd2h8AAAAJ&hl=en)
* [Kai Chen](https://scholar.google.co.jp/citations?user=TKvd_Z4AAAAJ&hl=en)

## Courses and Lectures

* [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/index.html)
* [Udacity Deep Learning](https://www.udacity.com/course/deep-learning--ud730)

## Datasets
### Training

* [Wikipedia](https://dumps.wikimedia.org/enwiki/)
* [WestburyLab.wikicorp.201004](http://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes)

### Evaluation

* [SemEval-2012 Task 2](https://www.cs.york.ac.uk/semeval-2012/task2.html)
* [WordSimilarity-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)
* [Stanford's Contextual Word Similarities (SCWS)](http://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes)
* [Stanford Rare Word (RW) Similarity Dataset](http://stanford.edu/~lmthang/morphoNLM/)

### Trained Word Vectors

* [Word2vec pretrained vector(English Only)](https://code.google.com/archive/p/word2vec/)
* [Word2vec pretrained vectors for 30+ languages](https://github.com/Kyubyong/wordvectors)
* [FastText pretrained vectors for 90 languages](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)
* [FastText pretrained vector for Japanese with NEologd](https://drive.google.com/open?id=0ByFQ96A4DgSPUm9wVWRLdm5qbmc)
* [word vectors trained by GloVe](http://nlp.stanford.edu/projects/glove/)
* [Dependency-Based Word Embeddings](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)
* [Meta-Embeddings](http://cistern.cis.lmu.de/meta-emb/)
* [Lex-Vec](https://github.com/alexandres/lexvec)
* [Huang et al. (2012)'s embeddings (HSMN+csmRNN)](http://stanford.edu/~lmthang/morphoNLM/)
* [Collobert et al. (2011)'s embeddings (CW+csmRNN)](http://stanford.edu/~lmthang/morphoNLM/)

## Articles

## Implementations and Tools
### Word2vec

* [Original](https://code.google.com/archive/p/word2vec/)
* [gensim](https://radimrehurek.com/gensim/models/word2vec.html)
* [TensorFlow](https://www.tensorflow.org/versions/r0.12/tutorials/word2vec/index.html)

### GloVe

* [Original](https://github.com/stanfordnlp/GloVe)
