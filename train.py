from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

path = '/Users/shangan/nlp/simsentence/similar-sentences/model/'
_vectorfile = 'vector.npy'
_trainset = 'VZ-SentenceUnique.txt'

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences = open(path+_trainset).read().splitlines()
sentence_embeddings = model.encode(sentences)
vecs = np.stack(sentence_embeddings)
print(vecs.shape)

model.save(path);
np.save(path+_vectorfile, sentence_embeddings) # save