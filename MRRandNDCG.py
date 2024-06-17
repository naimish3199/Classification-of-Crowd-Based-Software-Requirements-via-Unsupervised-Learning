import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
class MRR_NDCG:

    #### Mean Reciprocal Rank (MRR) ####
    def MRR(self,doc_term_matrix,vect,labels):
        dr= pd.DataFrame(doc_term_matrix)
        aa = cosine_distances(dr, dr)
        q = []
        c = 0
        for x in tqdm(range(len(aa))):
            dis = []
            la = []
            rank = 1
            for y in range(len(aa)):
                if x != y:
                    dis.append(aa[x][y])
                    la.append(labels[y])
            new = pd.DataFrame(list(zip(dis, la)),columns =['distance', 'labe'])  
            new = new.sort_values('distance')
            #print(new.head())
            for b in list(new['labe']):
                if labels[x] == b:
                    store = rank
                    break
                else:
                    rank = rank + 1
            q.append(1/store)          

        return round(np.mean(q),3)    

    #### Normalized Discounted Cumulative Gain (NDCG) ####            
    def NDCG(self,doc_term_matrix,vect,labels):
        dr= pd.DataFrame(doc_term_matrix)
        aa = cosine_distances(dr, dr)
        q = []
        for x in tqdm(range(len(aa))):
            dis = []
            la = []
            dcg = []
            idcg = []
            rank = 1
            k = 0
            r = 1
            j = 0
            rank = 1
            for y in range(len(aa)):
                if x != y:
                    dis.append(aa[x][y])
                    la.append(labels[y])
            new = pd.DataFrame(list(zip(dis, la)),columns =['distance', 'labe'])  
            new = new.sort_values('distance')
            
            newl = list(labels[0:x])+list(labels[x+1:])
            zz = list(new['labe'])
            b = 0
            while(b < len(zz) and j < len(newl)):
                if newl[j] == zz[b]:
                    dcg.append(float(1)/float(math.log2(rank+1)))
                rank = rank + 1
                b = b + 1
                j = j + 1
                
            b = 0    
            while(b < len(zz) and k < len(newl)):
                if newl[k] == zz[b]:
                    idcg.append(float(1)/float(math.log2(r+1)))
                    r = r + 1
                k = k + 1
                b = b + 1
            q.append(np.sum(dcg)/np.sum(idcg))
            
        return round(np.mean(q),3)        
        