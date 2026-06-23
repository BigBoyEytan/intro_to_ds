import pandas as pd
import numpy as np
from data import load_data, add_new_columns, data_analysis
from clustering import transform_data, kmeans, visualize_results


if __name__ == "__main__":
    # Part A
    data_path = "london.csv"
    print("Part A: ")
    df = load_data(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_transformed = add_new_columns(df)
    data_analysis(df_transformed)

    # Part B
    print()
    print("Part B: ")
    features = ['hum', 'wind_speed']
    kmeans_data = transform_data(df_transformed, features)
    for k in [3, 4, 6]:
        print(f"k = {k}")
        labels, centroids = kmeans(kmeans_data, k)
        print(np.array_str(centroids, precision=3, suppress_small=True))
        print()
        visualize_results(kmeans_data, labels, centroids, "plots.pdf")