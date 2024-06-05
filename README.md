# Classification of Crowd-Based Software Requirements via Unsupervised Learning

This repository contains code for the paper **Classification of Crowd-Based Software Requirements via Unsupervised Learning**


## Run Locally

* Clone the repository

    ```python
    git clone https://github.com/anonymous328084/Classification-of-Crowd-Based-Software-Requirements-via-Unsupervised-Learning.git
    ```

* Make sure that you are using Python 3.8+ version
* Go to the project directory
     ```python
    cd <path_to_directory_where_repository_is_cloned>
    ```
* Create a new virtual environment to avoid any dependencies issues
    ```python
    pip install virtualenv    # Installing virtualenv package
    virtualenv myenv          # myenv is new environment name
    .\myenv\Scripts\activate  # activating new environment

    ```
* Installing all libraries required for running codes
    ```python
    pip install -r requirements.txt
    ```
## Usage

As discussed in the paper, once we identified the top three embeddings for our dataset, for further experiments we proceed with the top 3 embeddings

* Embeddings used
    * SRoBERTa (https://huggingface.co/sentence-transformers/all-distilroberta-v1)
    * SBERT (https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
    * Word2Vec (Trained on CrowdRE dataset)

* Clustering used
    * K-means
    * HAC
    
* Labelling
    * Manual Labelling (Using BERTopic)
    * Automatic Labelling (Using Sematic Similarity)

* Number of clusters -> [2,3,4,5]

* For running experiments
    ```python
    python .\main.py
    ```
## Example

* For one of the configurations of manual labelling


    ```python
    python .\main.py
    ```

    ```powershell
    Enter 0 for manual labelling and 1 for automatic labelling ->
    Enter 0 or 1: 0
    Enter the type of embeddings to be used ->
    Enter 0 for Word2Vec(Self Trained), 1 for SBERT and 2 for SRoBERTa: 2
    Type of clustering ->
    Enter 0 for K-means clustering and 1 for Hierarchical agglomerative clustering (HAC): 1
    Would you like to merge Health and Other application domain ?
    Enter 1 for Yes and 0 for No: 1
    Enter the number of clusters ->
    Enter any integer value in the range of 2 to 4: 4
    
    
    health+other   energy entertainment safety
    
    
    cluster0->  home,water,smart,food,time,automatically,temperature,save,know,shower
    cluster1->  home,door,smart,know,house,pet,alert,time,lock,dog
    cluster2->  home,energy,light,room,save,turn,temperature,smart,automatically,house
    cluster3->  music,home,room,tv,voice,smart,house,play,turn,movie
    
    
    Enter 0 if cluster is health+other
    Enter 1 if cluster is energy
    Enter 2 if cluster is entertainment
    Enter 3 if cluster is safety
    
    
    Enter label of each cluster (starting from cluster0) seperated by a space: 0 3 1 2
    
    
       precision  recall  f1-score
    0      0.627   0.593     0.602

* For one of the configurations of automated labelling


    ```python
    python .\main.py
    ```

    ```powershell
    Enter 0 for manual labelling and 1 for automatic labelling ->
    Enter 0 or 1: 1
    Enter the type of embeddings to be used ->
    Enter 0 for Word2Vec(Self Trained): 0
    Type of clustering ->
    Enter 0 for K-means clustering and 1 for Hierarchical agglomerative clustering (HAC): 0
    Would you like to merge Health and Other application domain ?
    Enter 1 for Yes and 0 for No: 0
    Enter the number of clusters ->
    Enter any integer value in the range of 2 to 5: 2
    
    
    health   energy
       precision  recall  f1-score
    0      0.804   0.804     0.803
    
    health   entertainment
       precision  recall  f1-score
    0      0.803     0.8     0.801
    
    health   safety
       precision  recall  f1-score
    0      0.824   0.837     0.824
    
    health   other
       precision  recall  f1-score
    0      0.541   0.543     0.539
    
    energy   entertainment
       precision  recall  f1-score
    0      0.904     0.9     0.902
    
    energy   safety
       precision  recall  f1-score
    0      0.876   0.884     0.879
    
    energy   other
       precision  recall  f1-score
    0      0.802    0.82     0.799
    
    entertainment   safety
       precision  recall  f1-score
    0      0.861   0.878     0.868
    
    entertainment   other
       precision  recall  f1-score
    0      0.786   0.774     0.758
    
    safety   other
       precision  recall  f1-score
    0      0.665   0.691     0.623
    ```