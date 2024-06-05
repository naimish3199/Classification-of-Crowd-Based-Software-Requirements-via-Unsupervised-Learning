from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class automated:
    def automated_labelling(self,number,clustering,labels,embedd,corpus,compare,merged):

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
            vector_list = []
            selected_keys = []

            if number == 2:
                a = x[0]
                b = x[1]
                print("\n")
                t = tx[a] + "   " + tx[b]
                print(t)    
                selected_keys.append(tx[a])
                selected_keys.append(tx[b])
                subset_dict = dict((key, compare[key]) for key in selected_keys if key in compare) 
                vector_list.append(compare[tx[a]])
                vector_list.append(compare[tx[b]])
                new_lab = []
                new_w2vsf = []
                corpusx = []
                for i in range(len(labels)):
                    if labels[i] == a or labels[i] == b:
                        new_lab.append(labels[i])
                        new_w2vsf.append(embedd[i])
                        corpusx.append(corpus[i])
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
                selected_keys.append(tx[a])
                selected_keys.append(tx[b])
                selected_keys.append(tx[c])
                subset_dict = dict((key, compare[key]) for key in selected_keys if key in compare) 
                vector_list.append(compare[tx[a]])
                vector_list.append(compare[tx[b]])
                vector_list.append(compare[tx[c]])
                new_lab = []
                new_w2vsf = []
                corpusx = []
                for i in range(len(labels)):
                    if labels[i] == a or labels[i] == b or labels[i] == c:
                        new_lab.append(labels[i])
                        new_w2vsf.append(embedd[i])
                        corpusx.append(corpus[i])
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
                selected_keys.append(tx[a])
                selected_keys.append(tx[b])
                selected_keys.append(tx[c])
                selected_keys.append(tx[d])
                subset_dict = dict((key, compare[key]) for key in selected_keys if key in compare) 
                vector_list.append(compare[tx[a]])
                vector_list.append(compare[tx[b]])
                vector_list.append(compare[tx[c]])
                vector_list.append(compare[tx[d]])
      
                new_lab = []
                new_w2vsf = []
                corpusx = []
                for i in range(len(labels)):
                    if labels[i] == a or labels[i] == b or labels[i] == c or labels[i] == d:
                        new_lab.append(labels[i])
                        new_w2vsf.append(embedd[i])
                        corpusx.append(corpus[i])
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
                selected_keys.append(tx[a])
                selected_keys.append(tx[b])
                selected_keys.append(tx[c])
                selected_keys.append(tx[d])
                selected_keys.append(tx[e])
                subset_dict = dict((key, compare[key]) for key in selected_keys if key in compare) 
                vector_list.append(compare[tx[a]])
                vector_list.append(compare[tx[b]])
                vector_list.append(compare[tx[c]])
                vector_list.append(compare[tx[d]])
                vector_list.append(compare[tx[e]])
                new_lab = []
                new_w2vsf = []
                corpusx = []
                for i in range(len(labels)):
                    if labels[i] == a or labels[i] == b or labels[i] == c or labels[i] == d or labels[i] == e:
                        new_lab.append(labels[i])
                        new_w2vsf.append(embedd[i])
                        corpusx.append(corpus[i])
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
                    cluster1 = np.mean([new_w2vsf[x] for x in c1],axis=0)  
                    cluster2 = np.mean([new_w2vsf[x] for x in c2],axis=0)
                    c12345 = [c1,c2]
                    cluster12345 = [cluster1,cluster2]
                        
                if number == 3:
                    if y_pred[b] == 0:
                        c1.append(b)
                    elif y_pred[b] == 1:
                        c2.append(b)        
                    elif y_pred[b] == 2:
                        c3.append(b) 
                    cluster1 = np.mean([new_w2vsf[x] for x in c1],axis=0)  
                    cluster2 = np.mean([new_w2vsf[x] for x in c2],axis=0)
                    cluster3 = np.mean([new_w2vsf[x] for x in c3],axis=0)
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
                    cluster1 = np.mean([new_w2vsf[x] for x in c1],axis=0)  
                    cluster2 = np.mean([new_w2vsf[x] for x in c2],axis=0)
                    cluster3 = np.mean([new_w2vsf[x] for x in c3],axis=0)
                    cluster4 = np.mean([new_w2vsf[x] for x in c4],axis=0)
                    c12345 = [c1,c2,c3,c4]
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
                    cluster1 = np.mean([new_w2vsf[x] for x in c1],axis=0)  
                    cluster2 = np.mean([new_w2vsf[x] for x in c2],axis=0)
                    cluster3 = np.mean([new_w2vsf[x] for x in c3],axis=0)
                    cluster4 = np.mean([new_w2vsf[x] for x in c4],axis=0)
                    cluster5 = np.mean([new_w2vsf[x] for x in c5],axis=0)
                    c12345 = [c1,c2,c3,c4,c5] 
                    cluster12345 = [cluster1,cluster2,cluster3,cluster4,cluster5]                          

            pre = []
            rec = []
            f1 = []
            g = []

            def calculate_cosine_similarity_with_list(target_vector, vector_list):
                similarities = []
                for vector in vector_list:
                    target_vector = target_vector.reshape(1,-1)
                    vector = vector.reshape(1,-1)
                    similarity = cosine_similarity(target_vector, vector)[0][0]
                    similarities.append(similarity)
                return similarities

            if number == 2:
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster1, vector_list)))
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster2, vector_list)))
                
            if number == 3:
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster1, vector_list)))
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster2, vector_list)))
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster3, vector_list)))
                
            if number == 4:           
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster1, vector_list)))
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster2, vector_list)))
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster3, vector_list)))               
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster4, vector_list)))             

            if number == 5:           
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster1, vector_list)))
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster2, vector_list)))
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster3, vector_list)))               
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster4, vector_list)))   
                g.append(np.argmax(calculate_cosine_similarity_with_list(cluster5, vector_list)))   

            ypre = [0]*len(new_lab)
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
