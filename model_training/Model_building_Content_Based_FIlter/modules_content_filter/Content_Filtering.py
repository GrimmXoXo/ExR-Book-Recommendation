import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid
from modules_content_filter.Cleaning_content_filter import cleaning_dirty_data
import pickle



class ContentBasedRecommender:
    def __init__(self, book_data, word2vec_params=None, doc2vec_params=None,pickle_save=False):
        self.preprocessing_data = cleaning_dirty_data()
        self.book_data = book_data
        self.word2vec_params = word2vec_params if word2vec_params else {'vector_size': 100, 'window': 5, 'min_count': 1, 'workers': 4, 'epochs': 10}
        self.doc2vec_params = doc2vec_params if doc2vec_params else {'vector_size': 100, 'min_count': 1, 'epochs': 10}
        self.pickle_file = pickle_save
        self.embeddings=None

    def preprocessing(self, text):
        return self.preprocessing_data.text_cleaning(text)

    def pad_or_truncate(self, tokens, max_length, pad_token="<PAD>"):
        if len(tokens) > max_length:
            return tokens[:max_length]
        else:
            return tokens + [pad_token] * (max_length - len(tokens))

    def generate_embeddings(self):
        col = ['book_title', 'Genre_Interpretation', 'author', 'book_details']
        for column in col:
            self.book_data[f'{column}_tokens'] = self.book_data[column].apply(self.preprocessing)

        # Determine fixed length for padding/truncation
        lengths = [self.book_data[f'{column}_tokens'].apply(len) for column in col]
        fixed_length = int(np.percentile(np.concatenate(lengths), 90))
        print(f"Fixed length for padding/truncation: {fixed_length}")

        for column in col:
            self.book_data[f'{column}_tokens'] = self.book_data[f'{column}_tokens'].apply(lambda tokens: self.pad_or_truncate(tokens, fixed_length))

        all_tokens = self.book_data['book_title_tokens'].tolist() + self.book_data['genres_tokens'].tolist() + self.book_data['author_tokens'].tolist()
        
        word2vec_model = Word2Vec(sentences=all_tokens, **self.word2vec_params)

        # Generate Word2Vec embeddings for titles, genres, and authors
        self.book_data['title_embedding'] = self.book_data['book_title_tokens'].apply(lambda tokens: np.mean([word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv], axis=0))
        self.book_data['genres_embedding'] = self.book_data['genres_tokens'].apply(lambda tokens: np.mean([word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv], axis=0))
        self.book_data['author_embedding'] = self.book_data['author_tokens'].apply(lambda tokens: np.mean([word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv], axis=0))

        # Tagging documents for Doc2Vec
        tagged_data = [TaggedDocument(words=row['book_details_tokens'], tags=[str(row['book_id'])]) for index, row in self.book_data.iterrows()]

        # Train Doc2Vec model
        doc2vec_model = Doc2Vec(**self.doc2vec_params)
        doc2vec_model.build_vocab(tagged_data)
        doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

        # Generate Doc2Vec embeddings for book details
        self.book_data['details_embedding'] = self.book_data['book_id'].apply(lambda id: doc2vec_model.dv[str(id)])

        # Combine embeddings
        def combine_embeddings(row):
            title_emb = row['title_embedding']
            genres_emb = row['genres_embedding']
            author_emb = row['author_embedding']
            details_emb = row['details_embedding']
            combined_emb = np.concatenate((title_emb, genres_emb, author_emb, details_emb))
            return combined_emb

        self.book_data['combined_embedding'] = self.book_data.apply(combine_embeddings, axis=1)

        self.embeddings=self.book_data['combined_embedding']

        if self.pickle_file==True:
        # Save combined embeddings to a file using pickle
            with open('combined_embeddings.pkl', 'wb') as f:
                pickle.dump(self.book_data['combined_embedding'], f)

        # Calculate cosine similarities between book embeddings
        combined_embeddings_matrix = np.stack(self.book_data['combined_embedding'].values)
        cosine_similarities = cosine_similarity(combined_embeddings_matrix, combined_embeddings_matrix)
        return cosine_similarities

    # def get_recommendations(self, book_index, cosine_similarities, top_n=5):
    #     sim_scores = list(enumerate(cosine_similarities[book_index]))
    #     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #     sim_scores = sim_scores[1:top_n+1]  # Exclude the book itself
    #     book_indices = [i[0] for i in sim_scores]
    #     recommended_books = self.book_data.iloc[book_indices][['book_id', 'book_title']]
    #     return recommended_books

    # def get_recommendations_for_book(self, book_index=0, top_n=5):
    #     cosine_similarities = self.generate_embeddings()
    #     recommended_books = self.get_recommendations(book_index, cosine_similarities, top_n)
    #     return recommended_books
    

    def get_recommendations_for_book(self, book_title='', top_n=5):
        # Find the index corresponding to the given book title
        book_index = self.book_data[self.book_data['book_title'] == book_title].index[0]
        cosine_similarities = self.generate_embeddings()
        recommended_books = self.get_recommendations(book_index, cosine_similarities, top_n)
        return recommended_books

    def get_recommendations(self, book_index, cosine_similarities, top_n=5):
        sim_scores = list(enumerate(cosine_similarities[book_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]  # Exclude the book itself
        recommended_books = [(self.book_data.iloc[i]['book_title'], sim_scores[i][1]) for i in range(len(sim_scores))]
        return recommended_books



    def tune_word2vec(self, param_grid):
        best_params = None
        best_score = float('-inf')

        for params in ParameterGrid(param_grid):
            print(f'Training Word2Vec with params: {params}')
            # Preprocess the relevant columns
            self.book_data['book_title_tokens'] = self.book_data['book_title'].apply(self.preprocessing)
            self.book_data['genres_tokens'] = self.book_data['Genre_Interpretation'].apply(self.preprocessing)
            self.book_data['author_tokens'] = self.book_data['author'].apply(self.preprocessing)

            # Ensure consistent token lengths for concatenation
            all_tokens = self.book_data['book_title_tokens'].tolist() + self.book_data['genres_tokens'].tolist() + self.book_data['author_tokens'].tolist()

            model = Word2Vec(sentences=all_tokens, **params)
            # return model

            # Evaluate the model on some metric (e.g., cosine similarity)
            score = self.evaluate_model_word(model)
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
            self.book_data['book_details_tokens'] = self.book_data['book_details'].apply(self.preprocessing)

            tagged_data = [TaggedDocument(words=row['book_details_tokens'], tags=[str(row['book_id'])]) for index, row in self.book_data.iterrows()]
            
            model = Doc2Vec(**params)
            model.build_vocab(tagged_data)
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

            # Evaluate the model
            score = self.evaluate_model_doc(model)
            if score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score

    def evaluate_model_word(self, model):
        # Calculate embeddings for book titles, genres, and authors
        self.book_data['title_embedding'] = self.book_data['book_title_tokens'].apply(lambda tokens: np.mean([model.wv[token] for token in tokens if token in model.wv], axis=0))
        self.book_data['genres_embedding'] = self.book_data['genres_tokens'].apply(lambda tokens: np.mean([model.wv[token] for token in tokens if token in model.wv], axis=0))
        self.book_data['author_embedding'] = self.book_data['author_tokens'].apply(lambda tokens: np.mean([model.wv[token] for token in tokens if token in model.wv], axis=0))

        # Combine embeddings
        def combine_embeddings(row):
            title_emb = row['title_embedding']
            genres_emb = row['genres_embedding']
            author_emb = row['author_embedding']
            # Handle missing or empty embeddings
            if isinstance(title_emb, float) and np.isnan(title_emb):
                title_emb = np.array([])  # Replace NaN with an empty array
            
            if isinstance(genres_emb, float) and np.isnan(genres_emb):
                genres_emb = np.array([])  # Replace NaN with an empty array
            
            if isinstance(author_emb, float) and np.isnan(author_emb):
                author_emb = np.array([])  # Replace NaN with an empty array
            
            
            combined_emb = np.concatenate((title_emb, genres_emb, author_emb))
            return combined_emb

        self.book_data['combined_embedding'] = self.book_data.apply(combine_embeddings, axis=1)
        self.book_data['combined_embedding'] = self.book_data.apply(combine_embeddings, axis=1)

        # Pad or truncate embeddings to a fixed length
        max_length = max(len(embedding) for embedding in self.book_data['combined_embedding'])
        combined_embeddings_matrix = np.stack([np.pad(embedding, (0, max_length - len(embedding))) for embedding in self.book_data['combined_embedding'].values])

        # Compute cosine similarities
        # combined_embeddings_matrix = np.stack(self.book_data['combined_embedding'].values)
        cosine_similarities = cosine_similarity(combined_embeddings_matrix, combined_embeddings_matrix)

        # Set the diagonal values of the similarity matrix to 0 to ignore self-similarity
        np.fill_diagonal(cosine_similarities, 0)

        # Compute the average cosine similarity value
        average_similarity = cosine_similarities.mean()
        return average_similarity

    def evaluate_model_doc(self, model):
        # Generate Doc2Vec embeddings for book details
        self.book_data['details_embedding'] = self.book_data['book_id'].apply(lambda id: model.dv[str(id)])

        # Combine embeddings
        def combine_embeddings(row):
            details_emb = row['details_embedding']
            combined_emb = details_emb  # Assuming you want to evaluate details embedding only
            return combined_emb

        self.book_data['combined_embedding'] = self.book_data.apply(combine_embeddings, axis=1)

        # Compute cosine similarities
        combined_embeddings_matrix = np.stack(self.book_data['combined_embedding'].values)
        cosine_similarities = cosine_similarity(combined_embeddings_matrix, combined_embeddings_matrix)

        # Set the diagonal values of the similarity matrix to 0 to ignore self-similarity
        np.fill_diagonal(cosine_similarities, 0)

        # Compute the average cosine similarity value
        average_similarity = cosine_similarities.mean()
        return average_similarity
