import os
import zipfile

import numpy as np
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


def read_analogies(filename):
    """
    Reads through the analogy question file.

    Returns:
      questions: a [n, 4] numpy array containing the analogy question's word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(filename, 'rb') as analogy_f:
        for line in analogy_f:
            if line.startswith(b':'):  # Skip comments.
                continue
            words = line.strip().lower().split(b' ')
            ids = [word2id.get(w.strip()) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
    print('Eval analogy file: {}'.format(filename))
    print('Questions: {}'.format(len(questions)))
    print('Skipped: {}'.format(questions_skipped))
    analogy_questions = np.array(questions, dtype=np.int32)
    return analogy_questions

    
if __name__ == '__main__':
    """
    url = 'http://mattmahoney.net/dc/text8.zip'
    filename = maybe_download(url)
    unzip(filename)
    words = read_data(filename)
    print('Data size', len(words))
    """
    url = 'http://download.tensorflow.org/data/questions-words.txt'
    filename = maybe_download(url)
    analogy_questions = read_analogies(filename)
