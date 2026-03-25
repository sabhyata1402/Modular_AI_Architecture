#Methods related to data loading and all pre-processing steps will go here
import pandas as pd
from Config import Config

def get_input_data():
    df1 = pd.read_csv('Code_Architecture/data/AppGallery.csv',skipinitialspace=True, encoding='latin-1') 
    df2 = pd.read_csv('Code_Architecture/data/Purchasing.csv',skipinitialspace=True, encoding='latin-1')
    df = pd.concat([df1, df2], ignore_index=True)

        # Convert text columns to Unicode strings to ensure consistent encoding
    # across both datasets (handles any remaining non-ascii characters)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY]      = df[Config.TICKET_SUMMARY].values.astype('U')
    
    return df

# Remove duplicate content from email interactions within the same ticket.
def de_duplication(df):
    df = df.drop_duplicates()
    return df

def noise_remover(df):
    # Remove nulls in important columns
    df = df.dropna(subset=[Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT, 'Type 2'])
    return df

def translate_to_en(text_list):
    # Stub for translation. Actual translation can be added if required.
    return text_list

def drop_single_class_columns(df):
    # Type 1 has only one class, so model shouldn't predict it
    if Config.GROUPED in df.columns:
        df = df.drop(columns=[Config.GROUPED])
    return df
