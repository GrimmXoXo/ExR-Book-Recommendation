from sklearn.model_selection import ParameterGrid
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

class HyperparameterTuner:
    def __init__(self, book_data):
        """
        Initializes a new instance of the class with the given book data.

        Parameters:
            book_data (pandas.DataFrame): The book data to be used for training the model.

        Returns:
            None
        """
        self.book_data = book_data

    def tune_word2vec(self, param_grid):
        """
        Tunes the Word2Vec model using the given parameter grid.

        Parameters:
            param_grid (dict): A dictionary of parameter names and their respective values to be tuned.

        Returns:
            tuple: A tuple containing the best parameters and the best score obtained during the tuning process.
        """
        best_params = None
        best_score = float('-inf')

        for params in ParameterGrid(param_grid):
            print(f'Training Word2Vec with params: {params}')
            # Preprocess the relevant columns
            self.book_data['book_title_tokens'] = self.book_data['book_title'].apply(self.preprocess_text)
            self.book_data['genres_tokens'] = self.book_data['genres'].apply(self.preprocess_text)
            self.book_data['author_tokens'] = self.book_data['author'].apply(self.preprocess_text)

            all_tokens = self.book_data['book_title_tokens'].tolist() + self.book_data['genres_tokens'].tolist() + self.book_data['author_tokens'].tolist()
            model = Word2Vec(sentences=all_tokens, **params)

            # Evaluate the model on some metric (e.g., cosine similarity)
            score = self.evaluate_model(model)
            if score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score

    def tune_doc2vec(self, param_grid):
        best_params = None
        best_score = float('-inf')

        for params in ParameterGrid(param_grid):
            print(f'Training Doc2Vec with params: {params}')
            # Preprocess the relevant columns
            self.book_data['book_details_tokens'] = self.book_data['book_details'].apply(self.preprocess_text)

            tagged_data = [TaggedDocument(words=row['book_details_tokens'], tags=[str(row['book_id'])]) for index, row in self.book_data.iterrows()]
            model = Doc2Vec(**params)
            model.build_vocab(tagged_data)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

            # Evaluate the model on some metric (e.g., cosine similarity)
            score = self.evaluate_model(model)
            if score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score

    def preprocess_text(self, text):
        return word_tokenize(text.lower())

    def evaluate_model(self, model):
        book_embeddings = [model.dv[str(book_id)] for book_id in self.book_data['book_id']]
        cosine_similarities = cosine_similarity(book_embeddings)
        np.fill_diagonal(cosine_similarities, 0)  # Ignore self-similarity
        average_similarity = cosine_similarities.mean()
        return average_similarity 
