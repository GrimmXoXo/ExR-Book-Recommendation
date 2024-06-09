import os,sys
import pandas as pd
# # Add the project root to sys.path
# import modules_content_filter.Content_Filtering
# import modules_content_filter.Post_cleaning_Content_Filter
import modules_content_filter

print(sys.path)
class DataProcessor:
    def __init__(self):
        # self.data_path = data_path
        self.cleaning_module = modules_content_filter.Post_cleaning_Content_Filter.cleaning_dirty_data()      
        self.content_filtering = modules_content_filter.Content_Filtering

    def process_data(self):
        # Load data
        df = self.load_data()

        # Clean data
        cleaned_df = self.word_embed_cleaning(df)

        # Process data (e.g., word embedding)
        processed_data = self.process(cleaned_df)

        return processed_data

    def load_data(self):
        # Implement logic to load data from self.data_path
        df = pd.read_csv(self.data_path)
        df = df[~df.book_details.isna()]
        return df

    def process(self, df):
        # Implement data processing logic using the Content_based_filtering module
        # processed_data = self.content_filtering.ContentBasedRecommender(df)

        recommender = self.content_filtering.ContentBasedRecommender(df)
        param_grid_doc = {'vector_size': [100, 200], 'min_count': [1, 5], 'epochs': [10, 20]}
        param_grid_word={'vector_size': [100,200], 'window': [1, 5], 'min_count': [1,5], 'workers': [4]}
        best_params_word, best_score_word=recommender.tune_word2vec(param_grid_word)
        best_params_doc, best_score_doc=recommender.tune_doc2vec(param_grid_doc)

        # Get recommendations for the first book
        recommender=self.content_filtering.ContentBasedRecommender(df,word2vec_params=best_params_word,doc2vec_params=best_params_doc)
        recommendations = recommender.get_recommendations_for_book(book_title='The Amityville Horror', top_n=5)
        # print(recommendations)

        return recommendations

    def word_embed_cleaning(self,df):
        index_to_remove=set()
        cols=['author','book_title','book_details','Genre_Interpretation']
        for c in cols:
            a=df[c].apply(self.cleaning_module.text_cleaning)
            for i,v in a.items():
                if len(v)==0:
                    index_to_remove.add(i)

        df.drop(index_to_remove,inplace=True)
        df.reset_index(inplace=True,drop=True)


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description="Data Processing")
    # parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    # parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the model")

    # args = parser.parse_args()

    data_processor = DataProcessor()
    processed_data = data_processor.process_data()

    # Save the processed data or model
    # Implement logic to save the processed data or model to args.model_save_path
