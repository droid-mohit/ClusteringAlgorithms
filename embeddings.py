import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# Function to calculate the average vector for a document
def get_average_vector(doc, model):
    words = [word for word in doc if word in model.wv and word.strip()]  # Ignore blank values
    if not words:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[words], axis=0)


def sentence_transformers_embedding(input_df):
    print(f"total_size: {input_df.size}")
    model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(input_df['combined_text'].tolist(), convert_to_tensor=True)
    embeddings_array = embeddings.numpy()
    return embeddings_array


def word_2_vec_embedding(input_df):
    texts = input_df['combined_text']
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
    doc_vectors = [get_average_vector(doc, model) for doc in tokenized_texts if any(word.strip() for word in doc)]
    input_df['embedded_text'] = doc_vectors

    return input_df
