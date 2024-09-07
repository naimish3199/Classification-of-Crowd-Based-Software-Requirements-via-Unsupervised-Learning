# Classification of Crowd-Based Software Requirements via Unsupervised Learning

This repository contains the source code associated with the paper titled **Classification of Crowd-Based Software Requirements via Unsupervised Learning**. The code facilitates the reproduction of the experiments and results presented in the paper.

## Prerequisites
* Ensure that Python 3.8 or higher is installed on your system.

## Running the Project Locally
* **Clone the repository**

    ```bash
    git clone https://github.com/naimish3199/Classification-of-Crowd-Based-Software-Requirements-via-Unsupervised-Learning.git
    ```

* **Navigate to the Project Directory**
     ```bash
    cd <path_to_cloned_repository>
    ```
* **Set Up a Virtual Environment (to avoid any dependencies issues)**
  * Install the virtualenv package (if not already installed)
    ```bash
    pip install virtualenv   
    ```
  * Create a new virtual environment
    ```bash
    virtualenv myenv       
    ```
  * Activate the virtual environment
    
    On Windows
    ```bash
    .\myenv\Scripts\activate 
    ```
    On macOS/Linux
    ```bash
    source myenv/bin/activate
    ```

* **Install Required Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
## Usage
To evaluate the performance of various embedding techniques using Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG) metrics, execute the mrr_ndcg_calculation.ipynb notebook.

As outlined in the paper, after determining the top three performing embeddings on our dataset, subsequent experiments were conducted using these top embeddings.

* **Embeddings Utilized**

  The following embedding techniques were employed in the experiments:
    * **SRoBERTa** (https://huggingface.co/sentence-transformers/all-distilroberta-v1)
    * **SBERT** (https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
    * **Word2Vec** (Trained on CrowdRE dataset)

* **Clustering Techniques**

  The clustering methods used in this study include:
    * **K-means**
    * **Hierarchical Agglomerative Clustering (HAC)**
    
* **Labelling Methods**
  
  Labelling for the clusters was performed through both manual and automatic techniques:
  * **Manual Labelling**: Performed using **BERTopic**.
  * **Automatic Labelling**: Based on **Semantic Similarity**.

* **Number of clusters**
  
  The experiments were conducted with varying cluster numbers i.e. [2, 3, 4, 5].

* **Running the Experiments**

  To replicate the experiments, execute the following command
    ```python
    python .\main.py
    ```
## Example

* **Configuration 1: Manual Labelling**

  To execute the code for a specific configuration involving manual labelling, use the following command:

    ```bash
    python .\main.py
    ```
    Upon execution, the following prompts will be displayed:
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
    ```
    Sample output of clustering:
    ```bash    
    health+other   energy entertainment safety
    
    cluster0->  home,water,smart,food,time,automatically,temperature,save,know,shower
    cluster1->  home,door,smart,know,house,pet,alert,time,lock,dog
    cluster2->  home,energy,light,room,save,turn,temperature,smart,automatically,house
    cluster3->  music,home,room,tv,voice,smart,house,play,turn,movie
    ```
    You will then be prompted to assign labels to the clusters:
    ```bash
    Enter 0 if cluster is health+other
    Enter 1 if cluster is energy
    Enter 2 if cluster is entertainment
    Enter 3 if cluster is safety

    Enter label of each cluster (starting from cluster0) seperated by a space: 0 3 1 2
    ```
    The performance metrics (precision, recall, F1-score) for this configuration will be displayed as follows:
  ```bash
       precision  recall  f1-score
    0      0.627   0.593     0.602
  ```

* **Configuration 2: Automated Labelling**

    To execute the code for a specific configuration involving automated labelling, use the following command:

    ```bash
    python .\main.py
    ```
    Upon execution, the following prompts will be displayed:
    ```bash
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
    ```
    The performance metrics (precision, recall, F1-score) for this configuration will be displayed as follows:
  ```bash  
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
## Citation
