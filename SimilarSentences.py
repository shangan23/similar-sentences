import numpy as np
import scipy.spatial
import logging
from sentence_transformers import SentenceTransformer, LoggingHandler

class SimilarSentences():

    def __init__(self):

    def get_path(self):

    def train(self,sentences):
        path = self.get_path()
        vectorfile = 'vectorfile.npy'
        np.set_printoptions(threshold=100)
        logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        sentence_embeddings = model.encode(sentences)
        model.save(path);
        np.save(path+vectorfile, sentence_embeddings)

    def download_model(self):

    def check_file(self):
        
    def predict(self,similar_to,num):
        model = SentenceTransformer('/content/model_to_save')
        sentences_file = np.load('VzSentenceVectors.npy')
        vecs = np.stack(sentences_file)
        query_embeddings = model.encode(sentences_to_find)
        closest_n = num
        for query, query_embedding in zip(queries, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], vecs, "cosine")[0]
            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])
            print("\nQuery:", query)
            print("\nTop 5 most similar sentences in corpus:")
            for idx, distance in results[0:closest_n]:
                print(sentences[idx].strip(), "(Score: %.4f)" % (1-distance))