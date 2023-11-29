import pandas as pd
import numpy as np

from clustering_algorithms import run_dbscan_clustering, run_kmeans_clustering

file_path = "drd_alerts_list.csv"

embedding_column_names = ['source', 'title', 'text', 'timestamp', 'uuid']


def compile_text(x):
    text = (
        f"From: {x['source'].lower()} got alert of type: {x['title'].lower()} with error: {x['text']} "
        f"at: {x['timestamp']} with uuid: {x['uuid']} "
    )
    print(text)
    return text


def load_data():
    dataframe = pd.read_csv(file_path)
    dataframe = dataframe.dropna(axis=1, how='all')
    dataframe = dataframe.dropna(subset=embedding_column_names)
    dataframe['text'] = dataframe['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    dataframe['combined_text'] = dataframe.apply(lambda x: compile_text(x), axis=1)
    return dataframe


def dump_df_to_csv(dataframe):
    dataframe.to_csv('output_file.csv', index=False)


if __name__ == '__main__':
    df = load_data()

    # final_df = run_dbscan_clustering(df)
    # print(final_df[['source', 'title', 'text', 'cluster']])
    # dump_df_to_csv(final_df)

    final_df = run_kmeans_clustering(df)
    dump_df_to_csv(final_df)
