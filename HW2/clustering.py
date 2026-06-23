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
    prev_centroids = initial_centroids
    counter = 0
    consecutive = True

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

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', color='black')

    plt.xlabel('hum')
    plt.ylabel('wind_speed')

    plt.show()

    k = centroids.shape[0]
    plt.title(f'Results for kmeans with k = {k}')

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

    unique_labels = np.unique(labels)
    centroids =[]
    for unique_label in unique_labels:
        same_group = get_list_with_label_l1(data, labels, unique_label)
        new_centroid_of_group = get_new_centroid_of_group(same_group)
        centroids.append(new_centroid_of_group)

    return np.array(centroids)

def get_list_with_label_l1(data,labels,l1):
    list_of_elements_with_label_l1 =[]
    for index,element in enumerate(data):
        if labels[index] == l1:
            list_of_elements_with_label_l1.append(element)

    return list_of_elements_with_label_l1

def get_new_centroid_of_group(group):
    new_centroid_of_group = []

    sum_of_each_dim =[0.0,0.0]
    num_of_elements = group.shape[0]

    for element in group:
        for index,value in enumerate(element):
            sum_of_each_dim[index] += value

    new_centroid_of_group.append(sum_of_each_dim[0]/num_of_elements)
    new_centroid_of_group.append(sum_of_each_dim[1]/num_of_elements)

    return new_centroid_of_group