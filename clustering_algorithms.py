import csv

import pandas as pd
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from embeddings import sentence_transformers_embedding, word_2_vec_embedding


def join_and_handle_nan(row):
    return ' '.join(str(x) for x in row if pd.notna(x))


def run_kmeans_clustering(input_df):
    nltk.download("punkt")
    embeddings_array = sentence_transformers_embedding(input_df)

    # Use K-means to cluster the documents
    num_clusters = 6  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # doc_vectors = np.stack(input_df['embedded_text'].to_numpy())
    clusters = kmeans.fit_predict(embeddings_array)
    input_df['cluster'] = clusters

    # inertia_values = []
    # possible_num_clusters = range(1, 20)  # Adjust the range as needed
    # for num_clusters in possible_num_clusters:
    #     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    #     kmeans.fit(doc_vectors)
    #     inertia_values.append(kmeans.inertia_)
    #
    # # Plot the inertia values
    # plt.plot(possible_num_clusters, inertia_values, marker='o')
    # plt.title('Elbow Method for Optimal Number of Clusters')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    # plt.show()
    return input_df[['uuid', 'source', 'title', 'text', 'cluster']]


def run_dbscan_clustering(input_df):
    # # Extract relevant columns for clustering
    # text_columns = ['text']
    # data_for_clustering = input_df[text_columns]
    # data_for_clustering['concatenated_text'] = data_for_clustering.apply(join_and_handle_nan, axis=1)
    #
    # # Convert text data to numerical vectors using TF-IDF
    # vectorizer = TfidfVectorizer()
    # text_vectors = vectorizer.fit_transform(data_for_clustering['concatenated_text'])

    input_df = sentence_transformers_embedding(input_df)
    text_vectors = input_df['combined_text']
    # Optionally, reduce dimensionality using PCA
    pca = PCA(n_components=5)  # Adjust the number of components as needed
    text_vectors_pca = pca.fit_transform(text_vectors.toarray())

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
    clusters = dbscan.fit_predict(text_vectors_pca)
    input_df['cluster'] = clusters

    final_df = input_df[['uuid', 'source', 'title', 'text', 'cluster']]

    return final_df
