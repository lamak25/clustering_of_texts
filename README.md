# Clustering algorithm on text dataset

### Task:

#### CZ
(Context: D je primárně Data Govenance & Documentation platforma, ve které uživatelé píší dokumentaci k business pojmům, reportům, částem datových skladů, databázím, modelům, tabulkám, procedurám atd. Často se stavá, že uložené informace na zákaznické instalaci D je několik desítek a občas stovek GB dat textů. Pro orientaci je proto důležité mít data strukturovaná, ideálně nějak automaticky. Jedním ze způsobů je automatické seskupování objektů dle podobnosti a následné jejich tagování / labelování.)

> V přiloze najdete archiv, ve kterém jsou objekty a jejích attributy. Jeden objekt si lze představit jako jeden dokument. V jednom z objektů může být více attributů (např. název, description, summary atd). Úlohou je seskupit objekty dle jejích podobnosti pro jejich následující labelovaní. Počet skupin dopředu nevíme, tedy měl by se definovat automatický, iděálně pomocí nejakého algoritmu.

> Výsledkem by měli být skupiny objectů s ID jejich objektů. Formát výsledku může být libovolný. Může to být JSON, např. [“A”: [1,2,3], “B”: [5, 7, 9]], kde A a B jsou IDicka skupin. Nebo v formátu csv/tsv tabulky, kde budou sloupce Group_Id, Object_Id.

#### EN
(Context: D is primarily a Data Govenance & Documentation platform where users write documentation for business concepts, reports, parts of data warehouses, databases, models, tables, procedures, etc. It is often the case that the information stored on a customer installation of D is several tens and sometimes hundreds of GB of text data. It is therefore important to have the data structured for reference, ideally in some automatic way. One way is to automatically group objects by similarity and then tag/label them.)

> In the attachment you will find an archive containing the objects and their attributes. You can think of one object as one document. There can be multiple attributes in one of the objects (e.g. title, description, summary etc). The task is to group the objects according to its similarity for their subsequent labeling. We do not know the number of groups in advance, so it should be defined automatically, possibly by some algorithm.

> The result should be groups of objects with their object IDs. The format of the result can be arbitrary. It can be JSON, e.g. ["A": [1,2,3], "B": [5, 7, 9]], where A and B are the group IDs. Or in csv/tsv table format, where the columns will be Group_Id, Object_Id.

### Idea:

I divided the solution into 3 steps, where each step is implemented in its class and can be modified more or less on its own. The three steps are:

1. Load and clean the dataset 

2. Create embeddings

3. Cluster embeddings into groups

### HOW TO:
Install transformers and torch:

`pip3 install transformers`

`pip3 install torch`

Run `python3 main.py`

(There are some settings inside of the file main.py that can be eventually modified, such as input_filename, bert_model_name, result_filename, etc.)

### Solution & Discussion

#### 1. Load and clean the dataset 

I decided to simply take all the texts inside of the html tags and omit the rest of the tags. This can give fastly predictions on a good level, without the need to further processing them. In the future it could be eventually worth it to include for example svg tags as images, headline tags and perhaps even its parameters such as the size of the font, colour, etc. But including them smartly inside of our model would require more time and probably a different/more complicated model out of the scope of this project.

There are a few specifications while cleaning the dataset such as: 
* Delete words longer than 50 characters as that was some activation/security code.
* Some IDs were ignored if they didn't have any text documents or their document was only HTML tags without any text. 
* Documents were joined together if they had the same IDs.

Since in the next step, I will use the BERT model, there is no need for stemming and lemmatization

#### 2. Create embeddings

Now when I have cleaned text documents I need to convert them into vector/embedding so I can find similarities between them in the following step. I could use different models such as Bag of Words, TF-IDF or Doc2Vec. I chose a pre-trained [BERT](https://huggingface.co/bert-base-uncased "BERT model link") tokenizer and model which is the most advanced of them all and should give better results. (The tokenizer is responsible for tokenizing the input document and converting it into token IDs. The BERT model is then used to generate the document embeddings.)

Few of the documents resulted in tokens larger than 512 tokens. Since the BERT model I choose can process a maximum of 512 tokens I had to come up with a solution to process those documents. The options are:

* Ignore more tokens than the model allows (or even the whole document).
* Use a larger model.
* Split the tokens and feed the model with a subset of tokens and finally join them together.

I chose to split the tokens into multiple subsets and join them as an average of the vectors. This approach should keep as much information in the given document as possible.

#### 3 Cluster embeddings into groups

The final step is clustering the given embeddings into classes. The most commonly used models are K-means and Agglomerative Clustering (-> dendrogram). I implemented K-means which is less computationally intensive. Now I have to decide the number of clusters (k) to train K-means on. This can be done by testing several different k values and then with Elbow Methos choosing the best one.

As stated in [1](https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c "Elbow Method"):
_Elbow Method: Plot the number of clusters against the within-cluster sum of squares (WCSS), also known as the "inertia". The WCSS measures the compactness of the clusters. As the number of clusters increases, the WCSS typically decreases. Look for the "elbow" point in the plot, where the rate of decrease in WCSS starts to level off. This point represents a trade-off between the number of clusters and their compactness._

To automate the process of choosing the correct k value I calculated the so-called scaled inertia which gives a penalty for a higher number of classes as in [1](https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c "Elbow Method").

`Scaled_intertia = (Inertia(k)/Inertia(k=1)) + alfa*k`

* Inertia - a sum of the squared distance of samples to their closest cluster centre
* Alpha - manually tuned factor that gives a penalty to the number of clusters
* Inertia(k=1) - inertia for the basic situation in which all data points are in the same cluster

### Results

Resulted clusters are named 1...k, where k is the optimal number of clusters, and saved as JSON dictionary. Example: {"0":[ID1, ID2], "1" :[ID3, ID4]}