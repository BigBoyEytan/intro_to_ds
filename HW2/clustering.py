import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.00001^2)
    """
    noise = np.random.normal(loc=0, scale=1e-5, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


def load_data(path):
    """reads and returns the pandas DataFrame"""
    return pd.read_csv(path)


def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    selected_df = df[features]
    data_transformed = (selected_df - selected_df.min()) / (selected_df.max() - selected_df.min())
    return add_noise(data_transformed.to_numpy())


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    initial_centroids = choose_initial_centroids(data, k)
    curr_centroids = initial_centroids
    while True:
        prev_centroids = curr_centroids
        new_labels = assign_to_clusters(data, curr_centroids)
        curr_centroids = recompute_centroids(data, new_labels, k)
        if np.array_equal(prev_centroids, curr_centroids):
            break
    labels = np.array(new_labels)
    centroids = np.array(curr_centroids)
    return labels, centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', color='black', edgecolors='black', s=100)

    k = centroids.shape[0]
    plt.title(f'Results for kmeans with k = {k}')
    plt.xlabel('hum')
    plt.ylabel('wind_speed')

    plt.savefig(path)
    plt.close('all')


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the Euclidean distance
    """
    return np.sqrt(np.sum((x - y) ** 2))


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    labels = []
    for elemnt in data:
        min_dist = float('inf')
        closest_index = -1
        for i, centroid in enumerate(centroids):
            current_distance = dist(elemnt, centroid)
            if current_distance < min_dist:
                min_dist = current_distance
                closest_index = i
        labels.append(closest_index)
    return np.array(labels)


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    centroids =[]
    for unique_label in range(k):
        same_group = data[labels == unique_label]
        new_centroid_of_group = get_new_centroid_of_group(same_group)
        centroids.append(new_centroid_of_group)
    return np.array(centroids)


def get_new_centroid_of_group(group):
    """
        Computes the new centroid of a cluster by calculating the mean
        of all data points currently assigned to it.
        :param group: numpy array of shape (m, 2) containing the points in the cluster
        :return: list of 2 floats [mean_x, mean_y] representing the new centroid coordinates
        """
    return list(np.mean(group, axis=0))