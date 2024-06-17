import numpy as np
import pandas as pd
class pretrained:

    def avg_pretrained_embedding(self,model,corpus):
        """
        Computes the average sentence embedding for each sentence in a corpus using a pretrained model.
        """
        embed = []
        skipped_words = 0
        for x in corpus:
            sen = []
            for q in x:
                try:
                    sen.append(model[q])
                except:
                    for i in range(len(q)):
                        a = q[:i]
                        b = q[i:]
                        if a in model.index_to_key and b in model.index_to_key:
                            sum = (model[a] + model[b])/2
                            sen.append(sum)
                            break
                    skipped_words = skipped_words+1
                   
            embed.append(np.mean(sen,axis = 0))     
        # print(skipped_words)     
        return embed    
    
    
    def tfidf_embedding(self,model,corpus,tfidfarray,tflist):
        """
        Computes TF-IDF weighted sentence embeddings for each sentence in a corpus using a pretrained model.
        """
        embed = []
        skipped_words = 0
        count = 0
        for x in corpus:
            sen = []
            for q in x:
                try:
                    sen.append(model[q]*tfidfarray[count][tflist.index(q)])
                except:
                    for i in range(len(q)):
                        a = q[:i]
                        b = q[i:]
                        if a in model.index_to_key and b in model.index_to_key:
                            sum = (model[a] + model[b])/2
                            sen.append(sum*tfidfarray[count][tflist.index(q)])
                            break
                    skipped_words = skipped_words+1
                   
            embed.append(np.sum(sen,axis = 0))
            count = count + 1  
        return embed   