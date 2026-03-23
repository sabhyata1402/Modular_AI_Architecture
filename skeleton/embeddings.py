#Methods related to converting text in into numeric representation and then returning numeric representation may go here
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from Config import Config
import numpy as np
import random

# Fix random seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

# This file contains functions to convert email text into numeric vectors (embeddings).
# These functions are used in data_model.py to create the X matrix for modelling.

def get_tfidf_embd(df):
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90) # Only keep the 2000 most important words, ignore words that appear in fewer than 4 emails, and ignore words that appear in more than 90% of emails.
    text_data = df[Config.TICKET_SUMMARY].fillna('') + " " + df[Config.INTERACTION_CONTENT].fillna('') #     # Combine summary and interaction content into one text string per email for vectorisation
    X = tfidfconverter.fit_transform(text_data).toarray() #
    return X

# This function can be used to combine two different embedding matrices (e.g. from two different vectorisers) into one.
def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1) # Horizontally concatenate the two matrices (same number of rows, combine columns)