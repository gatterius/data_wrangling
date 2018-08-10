import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
import jellyfish  # for distance functions
from fuzzywuzzy import fuzz  # for distance functions
import numpy as np  # to process numeric arrays
from sklearn import random_projection

# calculate the distance between two given strings
def get_distance(string_a, string_b):
    # similarity scores given by edit distance functions are reversed  to turn them into distances
    lev = 1 - fuzz.ratio(string_a, string_b) / 100  # given value is normalized in range 1-100, not in 0-1
    jar = 1 - jellyfish.jaro_distance(string_a, string_b)
    jw = 1 - jellyfish.jaro_winkler(string_a, string_b)
    score = (lev + jar + jw) / 3  # calculate mean value of all distances
    return score


# drop the duplicates from the given cluster; a tuple is dropped if its similarity score with another tuple
# with same label is above the given threshold
def drop_duplicates_threshold(dataset_cluster, threshold):
    row_num = dataset_cluster.shape[0]
    for a in range(0, row_num):
        if a >= row_num:
            break
        row1 = dataset_cluster.iloc[a]
        for b in range(0, row_num):
            if a == b:
                continue
            if b >= row_num:
                break
            row2 = dataset_cluster.iloc[b]
            sim_sum = 0
            col_num = len(dataset_cluster.columns) - 1
            for i in range(0, col_num):
                sim_sum += 1-get_distance(str(row1[dataset_cluster.columns[i]]), str(row2[dataset_cluster.columns[i]]))
            score = col_num - sim_sum
            max_score = col_num - threshold*col_num
            if score <= max_score:
                dataset_cluster = dataset_cluster.drop(dataset_cluster.index[b])
                row_num -= 1
                b -= 1
                # row_num = dataset.shape[0]
    l = dataset_cluster.shape[0]

    return dataset_cluster

# vectorize dataset values, turning each tuple into the feature values using the Bag of Word approach
# Hashing vectorizing implements "feature hashing" technique: instead of building a hash table of the features
# encountered in training, as the vectorizers do, a hash function is applied to the features to determine their
# column index in sample matrices directly. It uses BoW for the initial feature extraction, but using the hashing
# trick allows to greatly optimize the performance, which makes this approach the best candidate to be used in the
# implementation of clustering workflow.
def vectorize_dataset(dataset):
    feature_matrix = []
    # define vectorizer
    vectorizer = HashingVectorizer(n_features=dataset.shape[1]*2)
    # iterate through all rows in the dataset
    for i in range(0, dataset.shape[0]):
        # extract row values
        row_values = list(dataset.iloc[i].astype(str))
        # vectorize the row
        vector = vectorizer.transform(row_values)
        # transform the created feature matrix from sparse to dense form
        dense_vector = vector.todense()
        # flatten the feature matrix, turning it into a single row
        dense_vector = np.array(dense_vector)
        flatten_vector = np.ndarray.flatten(dense_vector)
        # add feature row to the dataset feature matrix
        feature_matrix.append(list(flatten_vector))

    return feature_matrix


# cluster the veсtorized dataset using the Mean Shift algorithm
# The algorithms starts from initializing random seed and choosing the size of the window; the "center of mass" is
# detected by calculating the mean, and the search window is shifted towards this center; the process is repeated until
# convergence. For the clustering, the whole space is tessellated with the search windows, and all the point which are
# in the attraction basin (region where all trajectories lead to the same mode) belong to the same cluster
#
# prior to the clustering, the dimensionality of feature matrix is reduced using conventional PCA
# It identifies an "optimal" data projection to low-dimensional space, maximizing the data variance by finding new
# coordinate system. The method is based on the solving of the eigenvalue problem for the covariance matrix of the
# data (either calculating it directly or using Singular Value Decomposition in cases where the number of dimension
# is far greater than the number of data examples). The solution produces the eigenvectors and eigenvalues; the
# eigenvectors are ranked according to their eigenvalues (in the descending order of the eigenvalues), and top-N
# (called principal components) are chosen to be the coordinate axis of the new low-dimensional space. The matrix
# composed of these chosen eigenvectors is used on the data to transform it from high-dimensional to low-dimensional
# space.
def cluster_data(dataset, pca_comp, shift_quantile):
    cluster_data = vectorize_dataset(dataset)
    pca = PCA(n_components=pca_comp)
    pca.fit(cluster_data)
    pca_data = pca.transform(cluster_data)

    bandwidth = estimate_bandwidth(pca_data, quantile=shift_quantile, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(cluster_data)
    labels = ms.labels_
    label_df = pd.DataFrame(labels)
    new_data = dataset.join(label_df)
    label_col = list(new_data.columns)[len(new_data.columns)-1]
    new_data = new_data.rename(columns={label_col: 'label'})
    new_data.sort_values(by='label')

    return new_data


# drop the duplicates from the given clustered dataset, using the given distance threshold value
def drop_duplicates_clusters(dataset, distance_threshold):
    # extract labels, i.e. extract the clusters
    labels_unique = np.unique(dataset['label'])
    new_dataset = pd.DataFrame(columns=dataset.columns)
    # iterate through the clusters
    for label in labels_unique:
        data = dataset.loc[dataset['label'] == label]
        data = drop_duplicates_threshold(data, distance_threshold)
        new_dataset = new_dataset.append(data)


    return new_dataset

# Deduplication
# deduplicate data, using the defined functions
def deduplicate(merged_data, distance_threshold, pca_components_num, mean_shift_quantile):
    # drop all the identical duplicates just to reduce the computational expenses
    merged_data = merged_data.drop_duplicates(keep='first')
    # cluster data
    clustered_data = cluster_data(merged_data, pca_components_num, mean_shift_quantile)\
    # deduplicate clustered data
    deduped_data = drop_duplicates_clusters(clustered_data, distance_threshold)
    deduped_data = deduped_data.drop(columns='label')
    deduped_data = deduped_data[['provenance', 'postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]

    return deduped_data