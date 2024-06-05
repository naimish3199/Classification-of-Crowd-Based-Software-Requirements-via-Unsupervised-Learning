from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic

class manual_bertopic:
    def results(self,number,clustering,labels,embedd,corp,merged):

        if merged == False:
            tx = ['health', 'energy', 'entertainment', 'safety', 'other']
            if number == 2:
                ls = [[0, 1],[0, 2],[0, 3],[0, 4],[1, 2],[1, 3],[1, 4],[2, 3],[2, 4],[3, 4]]
            elif number == 3:
                ls = [[0,1,2],[0,1,3],[0,1,4],[1,2,3],[1,2,4],[2,3,4],[0,2,3],[0,2,4],[0,3,4],[1,3,4]]      
            elif number == 4:
                ls = [[0,1,2,3],[0,1,2,4],[0,1,3,4],[0,2,3,4],[1,2,3,4]]
            elif number == 5:
                ls = [[0,1,2,3,4]]
        
        elif merged == True:
            tx = ['health+other','energy', 'entertainment', 'safety']
            if number == 2:
                ls = [[0, 1],[0, 2],[0, 3],[1, 2],[1, 3],[2, 3]]
            elif number == 3:
                ls = [[0,1,2],[0,1,3],[1,2,3],[0,2,3]]        
            elif number == 4:
                ls = [[0,1,2,3]]   

        for x in ls:
            if number == 2:
                a = x[0]
                b = x[1]
                t = tx[a] + "   " + tx[b]
                print(t)   
                print('\n') 
                new_lab = []
                new_w2vsf = []
                corpx = []
                for i in range(len(labels)):
                    if labels[i] == a or labels[i] == b:
                        new_lab.append(labels[i])
                        new_w2vsf.append(embedd[i])
                        corpx.append(corp[i])
                for i in range(len(new_lab)):
                    if new_lab[i] == a:
                        new_lab[i] = 0
                    else:
                        new_lab[i] = 1

            elif number == 3:
                a = x[0]
                b = x[1]
                c = x[2]
                t = tx[a] + "   " + tx[b] + " " +  tx[c]
                print(t)
                print('\n')
                new_lab = []
                new_w2vsf = []
                corpx = []
                for i in range(len(labels)):
                    if labels[i] == a or labels[i] == b or labels[i] == c:
                        new_lab.append(labels[i])
                        new_w2vsf.append(embedd[i])
                        corpx.append(corp[i])
                for i in range(len(new_lab)):
                    if new_lab[i] == a:
                        new_lab[i] = 0
                    if new_lab[i] == b:
                        new_lab[i] = 1
                    if new_lab[i] == c:
                        new_lab[i] = 2           
                
            elif number == 4:
                print("\n")
                a = x[0]
                b = x[1]
                c = x[2]
                d = x[3]
                t = tx[a] + "   " + tx[b] + " " +  tx[c]+ " " + tx[d]
                print(t) 
                print('\n')
                new_lab = []
                new_w2vsf = []
                corpx = []
                for i in range(len(labels)):
                    if labels[i] == a or labels[i] == b or labels[i] == c or labels[i] == d:
                        new_lab.append(labels[i])
                        new_w2vsf.append(embedd[i])
                        corpx.append(corp[i])
                for i in range(len(new_lab)):
                    if new_lab[i] == a:
                        new_lab[i] = 0
                    if new_lab[i] == b:
                        new_lab[i] = 1
                    if new_lab[i] == c:
                        new_lab[i] = 2     
                    if new_lab[i] == d:
                        new_lab[i] = 3   

            elif number == 5:
                print("\n")
                a = x[0]
                b = x[1]
                c = x[2]
                d = x[3]
                e = x[4]
                t = tx[a] + "   " + tx[b] + " " +  tx[c]+ " " + tx[d] + " " + tx[e]
                print(t) 
                print('\n')
                new_lab = []
                new_w2vsf = []
                corpx = []
                for i in range(len(labels)):
                    if labels[i] == a or labels[i] == b or labels[i] == c or labels[i] == d or labels[i] == e:
                        new_lab.append(labels[i])
                        new_w2vsf.append(embedd[i])
                        corpx.append(corp[i])
                for i in range(len(new_lab)):
                    if new_lab[i] == a:
                        new_lab[i] = 0
                    if new_lab[i] == b:
                        new_lab[i] = 1
                    if new_lab[i] == c:
                        new_lab[i] = 2     
                    if new_lab[i] == d:
                        new_lab[i] = 3   
                    if new_lab[i] == e:
                        new_lab[i] = 4

            np.random.seed(42)
            p = []
            score = 0        
            if clustering == "kmeans":
                Kmean = KMeans(n_clusters=number, random_state=42)
                Kmean.fit(new_w2vsf)
                y_pred = Kmean.labels_                  
                
            elif clustering == 'hac':
                hierarchical_cluster = AgglomerativeClustering(n_clusters=number, affinity='euclidean', linkage='ward')
                y_pred = hierarchical_cluster.fit_predict(new_w2vsf) 
    
            c1 = []
            c2 = []
            c3 = []
            c4 = []
            c5 = []
            for b in range(len(y_pred)):
                if number == 2:
                    if y_pred[b] == 0:
                        c1.append(b)
                    elif y_pred[b] == 1:
                        c2.append(b)     
                    cluster1 = [corpx[x] for x in c1]
                    cluster2 = [corpx[x] for x in c2]
                    c12345 = [c1,c2]
                    cluster12345 = [cluster1,cluster2]
                        
                if number == 3:
                    if y_pred[b] == 0:
                        c1.append(b)
                    elif y_pred[b] == 1:
                        c2.append(b)        
                    elif y_pred[b] == 2:
                        c3.append(b) 
                    cluster1 = [corpx[x] for x in c1]
                    cluster2 = [corpx[x] for x in c2]
                    cluster3 = [corpx[x] for x in c3]
                    c12345 = [c1,c2,c3]
                    cluster12345 = [cluster1,cluster2,cluster3]        

                if number == 4:
                    if y_pred[b] == 0:
                        c1.append(b)
                    elif y_pred[b] == 1:
                        c2.append(b)        
                    elif y_pred[b] == 2:
                        c3.append(b)                        
                    elif y_pred[b] == 3:
                        c4.append(b)        
                    cluster1 = [corpx[x] for x in c1] # textual sentences
                    cluster2 = [corpx[x] for x in c2]
                    cluster3 = [corpx[x] for x in c3]
                    cluster4 = [corpx[x] for x in c4]
                    c12345 = [c1,c2,c3,c4] # indexes (list of list)
                    cluster12345 = [cluster1,cluster2,cluster3,cluster4]  

                if number == 5:
                    if y_pred[b] == 0:
                        c1.append(b)
                    elif y_pred[b] == 1:
                        c2.append(b)        
                    elif y_pred[b] == 2:
                        c3.append(b)                        
                    elif y_pred[b] == 3:
                        c4.append(b)       
                    elif y_pred[b] == 4:
                        c5.append(b)        
                    cluster1 = [corpx[x] for x in c1] # textual sentences
                    cluster2 = [corpx[x] for x in c2]
                    cluster3 = [corpx[x] for x in c3]
                    cluster4 = [corpx[x] for x in c4]
                    cluster5 = [corpx[x] for x in c5]
                    c12345 = [c1,c2,c3,c4,c5] 
                    cluster12345 = [cluster1,cluster2,cluster3,cluster4,cluster5]                                        
                        
            pre = []
            rec = []
            f1 = []

            ypre = [0]*len(new_lab)

            # Tokenizing topics
            vectorizer_model = CountVectorizer(stop_words="english")

            # Creating topic representation
            ctfidf_model = ClassTfidfTransformer()
            
            topic_model = BERTopic(      
              vectorizer_model=vectorizer_model, 
              ctfidf_model=ctfidf_model,          
              nr_topics=2
              )

            
            for j in range(number):
                
                dd = np.array([new_w2vsf[i] for i in c12345[j]]) #indexes
                topics, probs = topic_model.fit_transform(cluster12345[j],dd)
                lt = [wid for (wid, s) in topic_model.get_topic(0)]
                print("cluster{}".format(j) + "-> ", end = " ")
                for wrd in lt:
                    if wrd == lt[-1]:
                        print(wrd)
                    else:    
                        print(wrd,end =",")
            
            print('\n')

            if len(x) == 2:
                print("Enter 0 if cluster is {}".format(tx[x[0]]))
                print("Enter 1 if cluster is {}".format(tx[x[1]]))     
            if len(x) == 3:
                print("Enter 0 if cluster is {}".format(tx[x[0]]))
                print("Enter 1 if cluster is {}".format(tx[x[1]]))                
                print("Enter 2 if cluster is {}".format(tx[x[2]]))
            if len(x) == 4:
                print("Enter 0 if cluster is {}".format(tx[x[0]]))
                print("Enter 1 if cluster is {}".format(tx[x[1]]))                
                print("Enter 2 if cluster is {}".format(tx[x[2]]))
                print("Enter 3 if cluster is {}".format(tx[x[3]]))   
            if len(x) == 5:
                print("Enter 0 if cluster is {}".format(tx[x[0]]))
                print("Enter 1 if cluster is {}".format(tx[x[1]]))                
                print("Enter 2 if cluster is {}".format(tx[x[2]]))
                print("Enter 3 if cluster is {}".format(tx[x[3]]))   
                print("Enter 4 if cluster is {}".format(tx[x[4]]))

            
            print('\n')
            g = list(map(int, input("Enter label of each cluster (starting from cluster0) seperated by a space: ").split()))
            print('\n')
            for d in c1:
                ypre[d] = g[0]
            for d in c2:
                ypre[d] = g[1]
            if number == 3:
                for d in c3:
                    ypre[d] = g[2]    
            if number == 4:
                for d in c3:
                    ypre[d] = g[2]                            
                for d in c4:
                    ypre[d] = g[3]   

            if number == 5:
                for d in c3:
                    ypre[d] = g[2]                            
                for d in c4:
                    ypre[d] = g[3]  
                for d in c5:
                    ypre[d] = g[4]    

            pre.append(round(precision_score(new_lab, ypre, average='macro'),3))
            rec.append(round(recall_score(new_lab, ypre, average='macro'),3))
            f1.append(round(f1_score(new_lab, ypre, average='macro'),3))
            dictionary = {'precision':pre,'recall':rec,'f1-score':f1}
            bestdf = pd.DataFrame(dictionary)
            print(bestdf)
            print('\n')
