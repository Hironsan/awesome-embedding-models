# -*- coding: utf-8 -*-
import os
import zipfile

from keras.utils.data_utils import get_file


def maybe_download(url):
    """
    Download a file if not present.
    """
    filename = url.split('/')[-1]
    path = get_file(filename, url)
    return path


def read_data(filename):
    """
    Extract the first file enclosed in a zip file as a list of words.
    """
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data


def unzip(zip_filename):
    """
    Extract a file from the zipfile
    """
    with zipfile.ZipFile(zip_filename) as f:
        for filename in f.namelist():
            dirname = os.path.dirname(filename)
            f.extract(filename, dirname)
            return os.path.abspath(filename)


def read_analogies(filename, word2id):
    """
    Reads through the analogy question file.

    Returns:
      questions: a [n, 4] numpy array containing the analogy question's word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(filename, 'r') as analogy_f:
        for line in analogy_f:
            if line.startswith(':'):  # Skip comments.
                continue
            words = line.strip().lower().split()
            ids = [w in word2id for w in words]
            if False in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(words)
    print('Eval analogy file: {}'.format(filename))
    print('Questions: {}'.format(len(questions)))
    print('Skipped: {}'.format(questions_skipped))
    return questions

    
if __name__ == '__main__':
    url = 'http://mattmahoney.net/dc/text8.zip'
    filename = maybe_download(url)
    unzip(filename)
    words = read_data(filename)
    print('Data size', len(words))
    url = 'http://download.tensorflow.org/data/questions-words.txt'
    filename = maybe_download(url)
