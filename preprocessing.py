import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('punkt' , quiet=True)
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

class preprocessing:

    def processing(self,req):
        output = []
        corpus = []
        for i in range(len(req)):
            rev = re.sub('[^a-zA-Z]',' ', req[i])
            rev = rev.lower()
            corp = word_tokenize(rev) 
            words = [lemmatizer.lemmatize(x) for x in corp if x not in stop_words]
            corpus.append(words)    
        output.append(corpus)
        
        corp = []
        for j in range(len(corpus)):
            rev = ' '.join(corpus[j])
            corp.append(rev)
        output.append(corp)  
        
        allvocab = []
        for i in corpus:
            for j in i:
                allvocab.append(j)
        output.append(allvocab)
        
        freq = {}
        uniquevocab = list(set(allvocab))
        for x in uniquevocab:
            c = 0
            for y in allvocab:
                if x == y:
                    c = c + 1
            freq[x] = c/len(allvocab)        
        output.append(freq)
        
        wavg = []
        for x in corpus:
            a = 0
            for y in x:
                a = a + freq[y]
            wavg.append(a)   
        output.append(wavg)
        
        return output
