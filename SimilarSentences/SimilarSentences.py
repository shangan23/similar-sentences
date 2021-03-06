import numpy as np
import scipy.spatial
import logging
import json
import nltk
import zipfile
import os
import xlsxwriter
from sentence_transformers import SentenceTransformer, LoggingHandler
from .TrainSentences import TrainSentences
from sys import exit
from tqdm import tqdm

class SimilarSentences:

    def __init__(self, path, type):
        logging.basicConfig(level=logging.ERROR)
        nltk.download('punkt')
        if(type == "predict"):
            self.path = path
            self.load()
        elif(type == "train"):
            self.train_file = path
    
    #@todo: Method to be added in utility
    def get_file_extension(self, src):
        return os.path.splitext(src)[-1].lower()

    def load(self):
        dir_path = os.getcwd()
        model_path = dir_path+'/model/'
        self.model_path = model_path
        if(not os.path.isdir(model_path+'0_BERT') and not os.path.isdir(model_path+'1_Pooling') and not os.path.isfile(model_path+'vector.npy')):
            self.reload()
        else:
            print('\nFor reloading/updating the model try model.relaod()')

    def reload(self):
        dir_path = os.getcwd()
        model_path = dir_path+'/model/'
        print('\nScanning the path '+model_path + ' ...')
        path = self.path
        if(zipfile.is_zipfile(path)):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(model_path)
                if(os.path.isdir(model_path+'0_BERT') and os.path.isdir(model_path+'1_Pooling') and os.path.isfile(model_path+'vector.npy')):
                    print('Model validation OK...')
                else:
                    exit('Model file is not valid... exiting...')
        else:
            exit('Model file is not in .zip format')

    def get_path(self):
        _vector_file = 'vector.npy'
        _training_set = 'train.txt'
        _files = {
            'model': self.model_path,
            'vector': self.model_path + _vector_file,
            'training_set': self.model_path+_training_set
        }
        return _files

    def get_sentences(self, text):
        return nltk.sent_tokenize(text)

    def batch_predict(self, batch_file,num_of_simlar_sentences):
        dir_path = os.getcwd() + '/'
        file_path = dir_path + batch_file
        batch_output = []
        if(os.path.isfile(file_path) and self.get_file_extension(file_path) == ".txt"):
            print('Batch file validation OK...')
            batch_sentences = open(file_path).read().splitlines()
            with open(file_path) as bs: 
                sentences = bs.readlines()
                print("Batch prediction ...") 
                for sentence in tqdm(sentences): 
                    output = self.predict(sentence,num_of_simlar_sentences,"batch")
                    batch_output.append(output)
                self.create_excel(batch_output)                
                print("Working on Excel ...")
                print("Excel should be ready for usage ...")
                return batch_output
        else:
            exit('Batch file is not valid... exiting...')

    def predict(self, text, num_of_simlar_sentences, response="simple"):
        path = self.get_path()
        model = SentenceTransformer(path.get('model'))
        given_text = self.get_sentences(text)
        text_embedding = model.encode(given_text)
        return self.output(given_text, text_embedding, num_of_simlar_sentences, response)

    def output(self, text, embedding, closest_n, response):
        path = self.get_path()
        sentences = open(path.get('training_set')).read().splitlines()
        sentences_vector = np.load(path.get('vector'))
        vectors = np.stack(sentences_vector)
        output = {}

        i = 0
        while i < closest_n:
            out = []
            for query, sentence in zip(text, embedding):
                distances = scipy.spatial.distance.cdist(
                    [sentence], vectors, "cosine")[0]

                results = zip(range(len(distances)), distances)
                results = sorted(results, key=lambda x: x[1])
                index = 0
                for idx, distance in results[0:closest_n]:
                    if(i == index):
                        out.append(
                            {"sentence": sentences[idx].strip(), "score": (1-distance), "given":query})
                    index += 1
                output[i] = out
            i += 1
            
        if(response == "batch"):
            return output
        return self.json_output(output, response)

    def json_output(self, data, response):
        data = [*data.values()]
        json_out = ''
        if(response == "detailed"):
            json_out = json.dumps(data)
        elif(response == "simple"):
            simple_data = []
            for i in data:
                str = ''
                k = 0
                for j in i:
                    if(k == 0):
                        str += j['sentence']
                    else:
                        str += ' '+j['sentence']
                    k += 1
                simple_data.append(str)
            json_out = json.dumps(simple_data)
        return json_out

    def train(self, pretrain_model: str = None):
        model = TrainSentences(self.train_file, pretrain_model)
        model.train()

    def create_excel(self, data):
        print('\nGenerating Excel for data lenght {} '.format(len(data)))
        workbook = xlsxwriter.Workbook('Results.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, 'Sentence')
        worksheet.write(0, 1, 'Suggestion')
        worksheet.write(0, 2, 'Score')

        row = 1
        col = 0

        for o in tqdm(data):
            d = [*o.values()]
            for s in d:
                for l in s:
                    worksheet.write(row, col, l['given'])
                    worksheet.write(row, col + 1, l['sentence'])
                    worksheet.write(row, col + 2, l['score'])
                    row += 1

        workbook.close()
        print('Closing the workbook ...')