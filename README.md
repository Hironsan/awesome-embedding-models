# awesome-embedding-models[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of awesome embedding models tutorials, projects and communities.
Please feel free to pull requests to add links.

## Table of Contents


* **[Papers](#papers)**
* **[Researchers](#researchers)**
* **[Courses and Lectures](#courses-and-lectures)**
* **[Datasets](#datasets)**
* **[Implementations and Tools](#implementations-and-tools)**
<!--* **[Articles](#articles)**-->

## Papers
### Word Embeddings

**Word2vec, GloVe, FastText**

* Efficient Estimation of Word Representations in Vector Space (2013), T. Mikolov et al. [[pdf]](https://arxiv.org/pdf/1301.3781.pdf)
* Distributed Representations of Words and Phrases and their Compositionality (2013), T. Mikolov et al. [[pdf]](https://arxiv.org/pdf/1310.4546.pdf)
* word2vec Parameter Learning Explained (2014), Xin Rong [[pdf]](https://arxiv.org/pdf/1411.2738.pdf)
* word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method (2014), Yoav Goldberg, Omer Levy [[pdf]](https://arxiv.org/pdf/1402.3722.pdf)
* GloVe: Global Vectors for Word Representation (2014), J. Pennington et al. [[pdf]](http://nlp.stanford.edu/pubs/glove.pdf)
* Improving Word Representations via Global Context and Multiple Word Prototypes (2012), EH Huang et al. [[pdf]](http://www.aclweb.org/anthology/P12-1092)
* Enriching Word Vectors with Subword Information (2016), P. Bojanowski et al. [[pdf]](https://arxiv.org/pdf/1607.04606v1.pdf)
* Bag of Tricks for Efficient Text Classification (2016), A. Joulin et al. [[pdf]](https://arxiv.org/pdf/1607.01759.pdf)

**Language Model**

* Semi-supervised sequence tagging with bidirectional language models (2017), Peters, Matthew E., et al. [[pdf]](https://arxiv.org/abs/1705.00108)
* Deep contextualized word representations (2018), Peters, Matthew E., et al. [[pdf]](https://arxiv.org/abs/1802.05365)
* Contextual String Embeddings for Sequence Labeling (2018), Akbik, Alan, Duncan Blythe, and Roland Vollgraf. [[pdf]](http://alanakbik.github.io/papers/coling2018.pdf)
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018), [[pdf]](https://arxiv.org/abs/1810.04805)



**Embedding Enhancement**

* Sentence Embedding:Learning Semantic Sentence Embeddings using Pair-wise Discriminator(2018),Patro et al.[[Project Page]](https://badripatro.github.io/Question-Paraphrases/) [[Paper]](https://www.aclweb.org/anthology/C18-1230)
* Retrofitting Word Vectors to Semantic Lexicons (2014), M. Faruqui et al. [[pdf]](https://arxiv.org/pdf/1411.4166.pdf)
* Better Word Representations with Recursive Neural Networks for Morphology (2013), T.Luong et al. [[pdf]](http://www.aclweb.org/website/old_anthology/W/W13/W13-35.pdf#page=116)
* Dependency-Based Word Embeddings (2014), Omer Levy, Yoav Goldberg [[pdf]](https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf)
* Not All Neural Embeddings are Born Equal (2014), F. Hill et al. [[pdf]](https://arxiv.org/pdf/1410.0718.pdf)
* Two/Too Simple Adaptations of Word2Vec for Syntax Problems (2015), W. Ling[[pdf]](http://www.cs.cmu.edu/~lingwang/papers/naacl2015.pdf)


**Comparing count-based vs predict-based method**

* Linguistic Regularities in Sparse and Explicit Word Representations (2014), Omer Levy, Yoav Goldberg[[pdf]](https://www.cs.bgu.ac.il/~yoavg/publications/conll2014analogies.pdf)
* Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors (2014), M. Baroni [[pdf]](http://www.aclweb.org/anthology/P14-1023)
* Improving Distributional Similarity with Lessons Learned from Word Embeddings (2015), Omer Levy [[pdf]](http://www.aclweb.org/anthology/Q15-1016)


**Evaluation, Analysis**

* Evaluation methods for unsupervised word embeddings (2015), T. Schnabel [[pdf]](http://www.aclweb.org/anthology/D15-1036)
* Intrinsic Evaluation of Word Vectors Fails to Predict Extrinsic Performance (2016), B. Chiu [[pdf]](https://www.aclweb.org/anthology/W/W16/W16-2501.pdf)
* Problems With Evaluation of Word Embeddings Using Word Similarity Tasks (2016), M. Faruqui [[pdf]](https://arxiv.org/pdf/1605.02276.pdf)
* Improving Reliability of Word Similarity Evaluation by Redesigning Annotation Task and Performance Measure (2016), Oded Avraham, Yoav Goldberg [[pdf]](https://arxiv.org/pdf/1611.03641.pdf)
* Evaluating Word Embeddings Using a Representative Suite of Practical Tasks (2016), N. Nayak [[pdf]](https://cs.stanford.edu/~angeli/papers/2016-acl-veceval.pdf)

### Phrase, Sentence and Document Embeddings

**Sentence**

* [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726)
* [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx)
* [An efficient framework for learning sentence representations](https://arxiv.org/abs/1803.02893)
* [Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning](https://arxiv.org/abs/1804.00079)
* [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)

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

### Pre-Trained Language Models

Below is pre-trained [ELMo](https://arxiv.org/abs/1802.05365) models. Adding ELMo to existing NLP systems significantly improves the state-of-the-art for every considered task.

* [ELMo by AllenNLP](https://allennlp.org/elmo)
* [ELMo by TensorFlow Hub](https://alpha.tfhub.dev/google/elmo/2)

Below is pre-trained [sent2vec](https://github.com/epfml/sent2vec) models.
* [BioSentVec: sent2vec pretrained vector for biomedical text](https://github.com/ncbi-nlp/BioSentVec)

### Pre-Trained Word Vectors
Convenient downloader for pre-trained word vectors:
* [chakin](https://github.com/chakki-works/chakin)


Links for pre-trained word vectors:
* [Word2vec pretrained vector(English Only)](https://code.google.com/archive/p/word2vec/)
* [Word2vec pretrained vectors for 30+ languages](https://github.com/Kyubyong/wordvectors)
* [FastText pretrained vectors for 157 languages](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md)
* [FastText pretrained vector for Japanese with NEologd](https://drive.google.com/open?id=0ByFQ96A4DgSPUm9wVWRLdm5qbmc)
* [word vectors trained by GloVe](http://nlp.stanford.edu/projects/glove/)
* [Dependency-Based Word Embeddings](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/)
* [Meta-Embeddings](http://cistern.cis.lmu.de/meta-emb/)
* [Lex-Vec](https://github.com/alexandres/lexvec)
* [Huang et al. (2012)'s embeddings (HSMN+csmRNN)](http://stanford.edu/~lmthang/morphoNLM/)
* [Collobert et al. (2011)'s embeddings (CW+csmRNN)](http://stanford.edu/~lmthang/morphoNLM/)
* [BPEmb: subword embeddings for 275 languages](https://github.com/bheinzerling/bpemb)
* [Wikipedia2Vec: pretrained word and entity embeddings for 12 languages](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/)
* [word2vec-slim](https://github.com/eyaler/word2vec-slim)
* [BioWordVec: fastText pretrained vector for biomedical text](https://github.com/ncbi-nlp/BioSentVec)

<!--
## Articles
-->

## Implementations and Tools
### Word2vec

* [Original](https://code.google.com/archive/p/word2vec/)
* [gensim](https://radimrehurek.com/gensim/models/word2vec.html)
* [TensorFlow](https://www.tensorflow.org/versions/r0.12/tutorials/word2vec/index.html)

### GloVe

* [Original](https://github.com/stanfordnlp/GloVe)
* [GloVe as an optimized TensorFlow GPU Layer using chakin](https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer)

