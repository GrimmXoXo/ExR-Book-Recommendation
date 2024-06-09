import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import boxcox
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re


class cleaning_dirty_data():

    def __init__(self):
        

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        # print('loaded cleaning_data class')

    def text_cleaning(self,description):
        all_tokens=[]
        for desc in description:

            # Removing Emojis
            desc=self.__remove_emojis(desc)
            
            # Lowercase the text
            text = desc.lower()

            # Remove newlines
            text = text.replace('\n', ' ')
            
            # Tokenize the text
            tokens = word_tokenize(text)
            
            # Remove punctuation and special characters
            tokens = [word for word in tokens if word.isalnum()]
            
            # Remove stopwords
            tokens = [word for word in tokens if word not in self.stop_words]
            
            # Lemmatize the tokens
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

            all_tokens.append(tokens)
        return all_tokens
    
    def __remove_emojis(self,text):
        """
        Remove emojis from the given text.

        Parameters:
            text (str): The text from which emojis need to be removed.

        Returns:
            str: The text with emojis removed.
        """
        emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"  # dingbats
                "\U000024C2-\U0001F251" 
                "]+", 
                flags=re.UNICODE
            )            
        return emoji_pattern.sub(r'', text)


class preprocessing_data:
    
    def __init__(self):
        pass
    def missing_values_impute(self,df,col_name):
        if type(col_name)==list:
            for i in col_name:
                df[i]=df[i].fillna(df[i].median())
        else:
            df[col_name] = df[col_name].fillna(df[col_name].median())
        return df
    
    def remove_outliers(self,df,col_name):
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col_name] >= Q1 - 1.5 * IQR) & (df[col_name] <= Q3 + 1.5 * IQR)]
        return df
    def log_transformation(self,df,col_name):
        if type(col_name)==list:
            for i in col_name:
                df[i] = np.log(df[i] + 1)
        else:
            df[f'{col_name}'] = np.log(df[col_name] + 1)

    def sqrt_transformation(self,df,col_name):
        if type(col_name)==list:
            for i in col_name:
                df[i] = np.sqrt(df[i] + 1)
        else:
            df[f'{col_name}'] = np.sqrt(df[col_name] + 1)

    def boxcox_transformation(self,df,col_name):
        if type(col_name)==list:
            for i in col_name:
                df[f'{i}'], _ = boxcox(df[i] + 1)
        else:
            df[f'{col_name}'], _ = boxcox(df[col_name] + 1)

    def plot_histograms(self,df,col_names):
        # Plot histograms for each scaled column
        if type(col_names)==list:
            for col in col_names:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col], bins=20, kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.show()
        else:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col_names], bins=20, kde=True)
            plt.title(f'Distribution of {col_names}')
            plt.xlabel(col_names)
            plt.ylabel('Frequency')
            plt.show()
    def standard_scalar(self,df,col_name):
        if type(col_name)==list:
            for i in col_name:
                df[i] = (df[i] - df[i].mean()) / df[i].std()
        else:
            df[f'{col_name}'] = (df[col_name] - df[col_name].mean()) / df[col_name].std()

    @staticmethod
    def clean_text_bart(text):
    
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text.strip())  # Remove leading and trailing whitespaces
        
        return text






