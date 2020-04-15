import numpy as np
import logging
import zipfile
import os
from sentence_transformers import SentenceTransformer, LoggingHandler
from sys import exit

class TrainSentences:

    def __init__(self, txt_file):
        dir_path = os.getcwd() + '/'
        file_path = dir_path + txt_file
        print('Scanning the path '+file_path+ ' ...\n')
        if(os.path.isfile(file_path) and self.get_file_extension(file_path) == ".txt"):
            print('Training file validation OK...\n')
            self.train_file_path = file_path
            if not os.path.exists(dir_path+'trained_model'):
                os.makedirs(dir_path+'trained_model')
            self.model_save_path = dir_path+'trained_model/'
            self.zip_save_path = dir_path+'/'
        else:
            exit('Training file is not valid... exiting...')

    def get_file_extension(self,src):
        return os.path.splitext(src)[-1].lower()

    def get_path(self):
        _vector_file = 'vector.npy'
        _train_file = 'train.txt'
        _files = {
            'model': self.model_save_path,
            'vector': self.model_save_path + _vector_file,
            'training_set': self.train_file_path,
            'zip_path' : self.zip_save_path+'model.zip',
            'train_file' : self.model_save_path + _train_file,
        }
        return _files

    def train(self):
        path = self.get_path()
        np.set_printoptions(threshold=100)
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.ERROR,
                            handlers=[LoggingHandler()])
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        sentences = open(path.get('training_set')).read().splitlines()
        sentence_embeddings = model.encode(sentences)
        vecs = np.stack(sentence_embeddings)
        model.save(path.get('model'))
        print('\n')
        print('Saving the model to '+path.get('model')+'...\n')
        np.save(path.get('vector'), sentence_embeddings)
        print('Saving the vector to '+path.get('vector')+'...\n')
        print('Initiating model compression(.zip) ...\n')
        os.rename(path.get('training_set'), path.get('train_file'))
        self.compress_file(path.get('model'),path.get('zip_path'))
        print('~~~~~~~~~\n')
        print('Download model.zip and use it for prediction ...\n')
        print('~~~~~~~~\n')
        os.rmdir(self.model_save_path)

    def compress_file(self,dirpath, zippath):
        fzip = zipfile.ZipFile(zippath, 'w', zipfile.ZIP_DEFLATED)
        basedir = os.path.dirname(dirpath) + '/' 
        for root, dirs, files in os.walk(dirpath):
            dirname = root.replace(basedir, '')
            for f in files:
                fzip.write(root + '/' + f, dirname + '/' + f)
        fzip.close()