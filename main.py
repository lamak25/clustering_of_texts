from get_clean_data import getCleanData
from emb_model import embModel
from clustering import Clustering


input_filename = 'attributes.tsv' # input filename of documents
max_seq_length = 512  # Maximum sequence length for the embedding model
bert_model_name = 'bert-base-uncased' # Name of the embeddings model
file_path = 'doc_embeddings.json' # Path for saving embeddings
number_of_groups_img = 'Elbow_Method.png' # Img to show result of Elbow Method
result_filename = 'result_class.json' # Filename where to save resulted groups
alpha_k=0.009 # Penalty to number of clusters (smaller value -> more clusters)

print("Loading and cleaning data...")
gCD = getCleanData()
# Get all texts cleaned and ready for embedding model
all_texts = gCD.get_all_texts(input_filename)
total_texts = len(all_texts)
print("List length after deleting/joining duplicates and cleaning", total_texts)


print("Creating embeddings...")
eM = embModel()
# Change doc to emb
all_texts = eM.get_emb_from_docs(all_texts, bert_model_name, max_seq_length)
# Save embeddings
eM.save_embeddings(all_texts, file_path)


# Load embeddings
all_texts = eM.load_embeddings(file_path)
print("Clustering of embeddings...")

CS = Clustering(alpha_k)
# Prepare dataset
training_dataset, dataset_total = CS.get_training_dataset(all_texts)
# Get the best model and number of clusters
model, best_k = CS.get_best_model(training_dataset, dataset_total)
# Make predictions of the classes
predictions = model.predict(training_dataset)
# Format result
result_formated = CS.format_output(best_k, predictions, all_texts)
# print(result_formated)

# Save results as JSON and IMG
print("Saving results...")
CS.save_results(result_formated, result_filename)
CS.plot_K_values(best_k, number_of_groups_img)
print("DONE.")
