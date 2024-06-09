import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(data_path='bart_final_preprocess.csv'):
    # data_path = 'bart_final_preprocess.csv'
    df = pd.read_csv(data_path)
    df=df[~df.book_id.duplicated()]
    df.reset_index(inplace=True, drop=True)
    # Combine relevant features into a single string
    df['combined_features'] = df['book_title'] + ' ' + df['author'] + ' ' + df['genres'] + ' ' + df['book_details']
    df=df[~df['combined_features'].isna()]
    df.reset_index(inplace=True, drop=True)
    df.drop('Unnamed: 0',axis=1,inplace=True)
    # print(df['combined_features'])
    # # Use TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim,df


# Function to get book recommendations based on similarity
def get_recommendations(book_title):
    cosine_sim,df= preprocess()
    idx = df[df['book_title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:6]  # Top 5 recommendations including the book itself
    book_indices = [i[0] for i in sim_scores]
    return df['book_title'].iloc[book_indices], sim_scores
