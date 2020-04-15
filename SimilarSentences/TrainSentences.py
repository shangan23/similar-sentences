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
        print('Scanning the path '+file_path+ ' ...')
        if(os.path.isfile(file_path) and self.get_file_extension(file_path) == ".txt"):
            print('Training file validation OK...')
            self.train_file_path = file_path
            if not os.path.exists(dir_path+'trained_model'):
                os.makedirs(dir_path+'trained_model')
            self.model_save_path = dir_path+'trained_model/'
        else:
            exit('Training file is not valid... exiting...')

    def get_file_extension(src):
        retrun os.path.splitext(src)[-1].lower()

    def get_path(self):
        _vector_file = 'vector.npy'
        _files = {
            'model': self.model_save_path,
            'vector': self.model_save_path + _vector_file,
            'training_set': self.train_file_path
        }
        return _files

    def train(self):
        path = self.get_path()
        np.set_printoptions(threshold=100)
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO,
                            handlers=[LoggingHandler()])
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        sentences = open(path.get('training_set')).read().splitlines()
        sentence_embeddings = model.encode(sentences)
        vecs = np.stack(sentence_embeddings)
        model.save(path.get('model'))
        np.save(path.get('training_set'), sentence_embeddings)
        self.compress_file(path.get('model'))

    def compress_file(self, directory):
        # create a ZipFile object
        with ZipFile('model.zip', 'w') as zipObj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(directory):
                for filename in filenames:
                    # create complete filepath of file in directory
                    filePath = os.path.join(folderName, filename)
                    # Add file to zip
                    zipObj.write(filePath)
