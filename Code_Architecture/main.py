# Controller for the full pipeline
# Goes through each step, calling other modules as needed
#
#   1. load_data() - preprocess.get_input_data()
#   2. preprocess_data() - de_duplication() + noise_remover()
#   3. get_embeddings() - embeddings.get_tfidf_embd()
#   4. get_data_object() - Data(X, df), encapsulates all splits
#   5. perform_modelling() - modelling.model_predict()
# 
import numpy as np
import pandas as pd
import random

from preprocess import (
    get_input_data,
    de_duplication,
    noise_remover,
    translate_to_en
)
from embeddings import get_tfidf_embd
from modelling.modelling import model_predict
from modelling.data_model import Data
from Config import Config
from utils import time_it

# Set seeds
seed = 0
random.seed(seed)
np.random.seed(seed)

# Load raw data form the csvs
# Return combined dataframe from AppGallery and Purchasing
@time_it
def load_data() -> pd.DataFrame:
    return get_input_data()

# Clean raw data
#   de_duplication()
#   noise_remover()
#   translate_to_en()
@time_it
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = de_duplication(df)
    df = noise_remover(df)

    df[Config.TICKET_SUMMARY] = translate_to_en(
        df[Config.TICKET_SUMMARY].tolist()
    )
    df[Config.INTERACTION_CONTENT] = translate_to_en(
        df[Config.INTERACTION_CONTENT].tolist()
    )
    return df

# Convert email text into TF-IDF numeric feature matrix
@time_it
def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)
    return X, df

# Encapsulate all data into one Data object
# Builds all train test splits and chained label arrays
# Passes one object to all models
def get_data_object(X: np.ndarray, df: pd.DataFrame) -> Data:
    return Data(X, df)

# Train and evaluate all models with abstract interface
def perform_modelling(data: Data, name: str) -> None:
    model_predict(data, name)


# Entry
if __name__ == '__main__':

    # 1. Load raw data
    df = load_data()

    # 2. Clean and preprocess
    df = preprocess_data(df)

    # Process each group of Type 1 separately
    # Config.GROUPED splits dataframe into groups
    grouped_df = df.groupby(Config.GROUPED)

    for name, group_df in grouped_df:
        print(f"\n{'#'*55}")
        print(f"    Processing group : {name.strip()}")
        print(f"{'#'*55}")

        # 3. Vectorise text for this group
        X, group_df = get_embeddings(group_df)

        # 4. Build Data object with all splits and chained labels
        data = get_data_object(X, group_df)

        # 5. Train and evaluate all models
        perform_modelling(data, name.strip())