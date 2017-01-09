import os
import zipfile

from keras.utils.data_utils import get_file

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename):
    """
    Download a file if not present.
    """
    path = get_file(filename, origin=url + filename)
    return path


def read_data(filename):
    """
    Extract the first file enclosed in a zip file as a list of words.
    """
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

"""
def unzip(filename):
    dirname = os.path.dirname(filename)
    with zipfile.ZipFile(filename) as f:
        f.extractall(dirname)
"""

def unzip(zip_filename):
    with zipfile.ZipFile(zip_filename) as f:
        for filename in f.namelist():
            dirname = os.path.dirname(filename)
            f.extract(filename, dirname)
            return os.path.abspath(filename)

if __name__ == '__main__':
    filename = maybe_download('text8.zip')
    unzip(filename)
    words = read_data(filename)
    print('Data size', len(words))
