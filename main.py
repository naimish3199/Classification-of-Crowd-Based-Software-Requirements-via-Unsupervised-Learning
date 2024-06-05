import nltk
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
warnings.filterwarnings("ignore")

from preprocessing import *
from automated import *
from manual_bertopic import *

preprocess = preprocessing()
automatic_labelling = automated()
manual_labelling = manual_bertopic()

data = pd.read_csv("data/requirements.csv")
data['requirements'] = data['feature']+ ", " + data['benefit'] + '.'
d = pd.DataFrame(list(zip(data['requirements'], data['application_domain'])),columns = ['requirements','class'])
d['n_class'] = d['class']
# replacing values
d['n_class'].replace(['Health', 'Energy', 'Entertainment', 'Safety', 'Other'],[0,1,2,3,4], inplace=True)
labels = d['n_class']
namelabels = data['application_domain']

merged_label = []
for r in labels:
    if r == 4:
        merged_label.append(0)
    else:
        merged_label.append(r)

#Pre Processing
corpus,corp,allvocab,freq,wavg = preprocess.processing(d['requirements'])

sbert = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
s_embeddings = sbert.encode(corp)


word2vec = Word2Vec(corpus, min_count = 1,vector_size = 100,window = 5,sg = 1,epochs=30, seed = 1) # 1-> skipgram, 0-> cbow
avgword2vec = []
for x in corpus:
    avgword2vec.append(np.mean([word2vec.wv[token] for token in x if token in word2vec.wv.index_to_key],axis=0))

sroberta = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
sroberta_emb = sroberta.encode(corp)

def compare_dictionary(embeddings):

    health = []
    energy = []
    entertainment = []
    safety = []
    other = []

    for i in range(len(embeddings)):
        if namelabels[i] == 'Health':
            health.append(embeddings[i])
        elif namelabels[i] == 'Energy':
            energy.append(embeddings[i])
        elif namelabels[i] == 'Entertainment':
            entertainment.append(embeddings[i])
        elif namelabels[i] == 'Safety':
            safety.append(embeddings[i])
        elif namelabels[i] == 'Other':
            other.append(embeddings[i])

    compare = {}
    compare['health'] = np.mean(health, axis = 0)
    compare['energy'] = np.mean(energy, axis = 0)
    compare['entertainment'] = np.mean(entertainment, axis = 0)
    compare['safety'] = np.mean(safety, axis = 0)
    compare['other'] = np.mean(other, axis = 0)
    health_other = health + other
    compare['health+other'] = np.mean(health_other, axis = 0)

    return compare

print("Enter 0 for manual labelling and 1 for automatic labelling -> ")
user = int(input("Enter 0 or 1: "))
if user not in [0, 1]:
    print("Wrong Input")
    exit(0)

print("Enter the type of embeddings to be used -> ")
if user == 0:
    embedding = int(input("Enter 0 for Word2Vec(Self Trained), 1 for SBERT and 2 for SRoBERTa: "))
    embeddings_dict = {0: avgword2vec, 1: s_embeddings, 2: sroberta_emb}

elif user == 1:
    embedding = int(input("Enter 0 for Word2Vec(Self Trained): "))
    embeddings_dict = {0: avgword2vec}

embedding = embeddings_dict.get(embedding)
compare = compare_dictionary(embedding)

print("Type of clustering -> ")
clustering = int(input("Enter 0 for K-means clustering and 1 for Hierarchical agglomerative clustering (HAC): "))
clustering_dict = {0: 'kmeans', 1: 'hac'}
clustering = clustering_dict.get(clustering)
if clustering is None:
    print("Wrong Input")
    exit(0)

print("Would you like to merge Health and Other application domain ?")
merge =  int(input("Enter 1 for Yes and 0 for No: ").strip()) == 1


print("Enter the number of clusters -> ")
if merge == True:
    labels = merged_label
    number_of_clusters = int(input("Enter any integer value in the range of 2 to 4: "))
else:
    number_of_clusters = int(input("Enter any integer value in the range of 2 to 5: "))

if number_of_clusters < 2 & number_of_clusters > 5:
    print("Wrong Input")
    exit(0)


if user == 0:
    manual_labelling.results(number_of_clusters,clustering,labels,embedding,corp, merge)
elif user == 1:
    automatic_labelling.automated_labelling(number_of_clusters,clustering,labels,embedding,corpus,compare, merge)
else:
    print("Wrong Input")
    exit(0)
   
