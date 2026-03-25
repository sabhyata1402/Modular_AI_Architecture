from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from Config import Config
import numpy as np
import random

# Fix random seed
seed = 0
random.seed(seed)
np.random.seed(seed)

# Functions to convert email text into numeric vectors (embeddings)
# Used in data_model.py to create the X matrix for modelling

def get_tfidf_embd(df):
    # Only keep the 2000 most important words
    # Ignore words that appear in fewer than 4 emails, ignore words that appear in more than 90% of emails
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    # Combine summary and interaction content into one text string per email
    text_data = df[Config.TICKET_SUMMARY].fillna('') + " " + df[Config.INTERACTION_CONTENT].fillna('') 
    X = tfidfconverter.fit_transform(text_data).toarray() #
    return X

# Combine two different embedding matrices
def combine_embd(X1, X2):
    # Concat the two matrices
    return np.concatenate((X1, X2), axis=1)