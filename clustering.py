
import math
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler

# Useful resources: https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c

# Inertia is the sum of squared distance of samples to their closest cluster center. We would like this number to be as small as possible.

class Clustering:

    def __init__(self, alpha_k=0.009):
        self.scaled_inertia = []
        self.inertia = []
        self.alpha_k = alpha_k

    def train_model_get_inertia(self, training_dataset, k):
        # Evalueate given K value
        inertia_o = np.square((training_dataset - training_dataset.mean(axis=0))).sum()
        # fit k-means
        kmeans = KMeans(n_clusters=k, random_state=0).fit(training_dataset)
        scaled_inertia = kmeans.inertia_ / inertia_o + self.alpha_k * k
        return scaled_inertia, kmeans.inertia_

    def select_correct_k(self, training_dataset, k_range):
        # Try all range of K's to select the best k-value
        scaled_inertia = []
        inertia = []
        for k in k_range:
            scaled_inertia1, inertia1 = self.train_model_get_inertia(training_dataset, k)
            scaled_inertia.append((k, scaled_inertia1))
            inertia.append((k, inertia1))

        self.scaled_inertia = pd.DataFrame(scaled_inertia, columns = ['k','Scaled_Inertia']).set_index('k')
        self.inertia = pd.DataFrame(inertia, columns = ['k','Inertia']).set_index('k')
        best_k = self.scaled_inertia.idxmin()[0]
        return best_k

    def get_best_model(self, training_dataset, dataset_total):
        # Return the best model with the most suitable number of cluster

        # Prepare range for testing the optiomal K value
        k_range = range(2, int(math.sqrt(dataset_total)))
        best_k = self.select_correct_k(training_dataset, k_range)

        print("Optimal number of clusters:", best_k)
        # Get the best model and train it again
        kmeans = KMeans(n_clusters=best_k, random_state=0).fit(training_dataset)
        return kmeans, best_k

    def get_training_dataset(self, all_texts):
        # Take only embeddings without ID and scale the datasets
        training_dataset = []
        for i in all_texts:
            training_dataset.append(i[1])

        dataset_total = len(training_dataset)
        training_dataset = np.array(training_dataset)

        # create data matrix
        training_dataset = np.asarray(training_dataset).astype(float)
        # scale the data
        mms = MinMaxScaler()
        training_dataset = mms.fit_transform(training_dataset)

        return training_dataset, dataset_total

    # Format output
    def format_output(self, best_k, predictions, all_texts):
        # Put the result into different format

        # Prepare dictionary with number of classes
        result_formated = dict()
        for i in range(best_k):
            result_formated[str(i)] = []

        # Put the IDs into corresponding class
        for i, item in enumerate(all_texts):
            result_formated[str(predictions[i])].append(item[0])
        return result_formated

    def save_results(self, result_formated, file_path):
        # Export the array as JSON into a file
        with open(file_path, 'w+') as output_file:
            json.dump(result_formated, output_file)

    def plot_K_values(self, best_k, save_as_image = False):
        # Plot the inertia&scaled_inertia values and mark the knee point
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Scaled Inertia', color=color)
        ax1.plot(range(2, len(self.scaled_inertia)+2), self.scaled_inertia, color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Inertia', color=color)  # we already handled the x-label with ax1
        ax2.plot(range(2, len(self.inertia)+2), self.inertia, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.axvline(x=best_k, color='r', linestyle='--', label='Optimal Number of Clusters')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend()
        plt.show()

        # Save the plot as an image file (e.g., PNG)
        if save_as_image:
            plt.savefig(save_as_image)

# Testing dataset:
# all_texts = [[123,[1,2,3,4,5]], [654,[8,7,9,6,5]], [789,[1,5,7,3,6]], [741,[7,8,4,1,2]], [852,[3,2,5,4,9]], [963,[0,2,5,5,8]], [128,[1,0,3,4,0]], [657,[8,0,9,6,0]], [759,[0,5,7,3,0]], [701,[7,0,4,0,2]], [850,[3,0,0,4,0]], [999,[0,0,0,5,8]], [129,[1,2,7,7,5]], [659,[7,7,9,6,5]], [799,[7,5,7,7,6]], [798,[7,8,7,1,7]],]
