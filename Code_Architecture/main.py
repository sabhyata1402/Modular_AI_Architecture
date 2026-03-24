# =============================================================================
# main.py
# =============================================================================
# PURPOSE:
#   Main controller and entry point for the pipeline.
#   Orchestrates every step — contains NO implementation logic itself.
#
# ARCHITECTURAL PRINCIPLES:
#   Separation of Concerns (Feature 1) — only orchestrates, never implements.
#   Abstraction (Feature 3) — calls model.train/predict/print_results without
#   knowing which model it is running.
#
# PIPELINE:
#   1. load_data()                 → preprocess.get_input_data()
#   2. preprocess_data()           → de_duplication() + noise_remover()
#                                    + translate_to_en() (stub)
#   3. get_embeddings()            → embeddings.get_tfidf_embd()
#   4. drop_single_class_columns() → preprocess.drop_single_class_columns()
#   5. get_data_object()           → modelling.Data(X, df)
#   6. perform_modelling()         → modelling.model_predict()
#
# TEAMMATE: Both
# =============================================================================


from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

# Step 1 : Load raw data from both CSV files.
# Calls preprocess.get_input_data() which loads AppGallery.csv and
# Purchasing.csv, concatenates them, and converts text to Unicode.

def load_data():
    df = get_input_data()
    return  df

#Step 2: Clean the raw email data.
#     de_duplication(df)   — removes exact duplicate rows
#    noise_remover(df)    — drops rows with nulls in key columns
#    translate_to_en()    — stub: returns text unchanged. Full implementation would translate German,Portuguese, Italian etc. into English using stanza + facebook/m2m100_418M model.

def preprocess_data(df):
  
    df =  de_duplication(df)
    df = noise_remover(df)

    # Translation stub — called on both text columns.
    # Returns text unchanged until a real translation model is integrated.
    df[Config.TICKET_SUMMARY]      = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    df[Config.INTERACTION_CONTENT] = translate_to_en(df[Config.INTERACTION_CONTENT].tolist())
    return df

# Step 3 : Convert email text to TF-IDF numeric feature matrix. Calls embeddings.get_tfidf_embd() — combines Ticket Summary + Interaction content and produces X of shape (n_emails, 2000).
def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  
    return X, df

# Step 4 : Encapsulate all data into one Data object (Encapsulation). Builds X_train, X_test, y_train, y_test, y23_train, y23_test, y234_train, y234_test from the TF-IDF matrix and cleaned DataFrame.
def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

# Run all models via uniform abstract interface (Abstraction). Calls modelling.model_predict() which runs RandomForest (baseline) then ChainedRandomForest (CA1 Design Choice 1).
def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

# Code will start executing from following line
if __name__ == '__main__':
    
    # Step 1: Load
    df = load_data()

    # Step 2: Preprocess
    df = preprocess_data(df)

    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(f"  Processing group: {name.strip()}")
    
        # Step 4 : Drop single-class columns to avoid modelling errors. (Design Choice 2)
        X, group_df = get_embeddings(group_df)
    
        # Step 5: Build Data object (encapsulates all train/test arrays)
        data = get_data_object(X, group_df)
        
        # Step 6: Train and evaluate all models
        perform_modelling(data, group_df, 'name')

