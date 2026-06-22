import numpy as np

from HW2.clustering import *
from HW2.data import *

data = load_data('/Users/eytangorenshtein/PycharmProjects/Intro_Data_Science_folder/HW2/london.csv')

for k in [3, 4, 6]:
    labels, centroids = kmeans(data, k)

    print(f"k = {k}")
    print(np.array_str(centroids, precision=3, suppress_small=True))
    print()

    path_to_save = f"plot_k_{k}.png"
    visualize_results(data, labels, centroids, path_to_save)