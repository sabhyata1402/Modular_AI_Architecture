#This is a main file: The controller. All methods will directly on directly be called here
from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame, target_series: pd.Series):
    return Data(X, df, target_series)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

# Code will start executing from following line
if __name__ == '__main__':
    
    # pre-processing steps
    df = load_data()
    df = preprocess_data(df)
    df = drop_single_class_columns(df)
    
    # Fill NAs in types to allow clean string concatenation
    for col in Config.TYPE_COLS:
        df[col] = df[col].fillna('')
        
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # data transformation
    X, group_df = get_embeddings(df)
    
    # Design Choice 1: Chained Multi-Outputs
    
    # Target 1: Type 2
    t1_series = df[Config.TYPE_COLS[0]]
    data_t1 = get_data_object(X, df, t1_series)
    perform_modelling(data_t1, df, 'Chained_Target_1 (Type 2)')
    
    # Target 2: Type 2 + Type 3
    t2_series = df[Config.TYPE_COLS[0]] + ' ' + df[Config.TYPE_COLS[1]]
    t2_series = t2_series.str.strip()
    data_t2 = get_data_object(X, df, t2_series)
    perform_modelling(data_t2, df, 'Chained_Target_2 (Type 2 + Type 3)')
    
    # Target 3: Type 2 + Type 3 + Type 4
    t3_series = df[Config.TYPE_COLS[0]] + ' ' + df[Config.TYPE_COLS[1]] + ' ' + df[Config.TYPE_COLS[2]]
    t3_series = t3_series.str.strip()
    data_t3 = get_data_object(X, df, t3_series)
    perform_modelling(data_t3, df, 'Chained_Target_3 (Type 2 + Type 3 + 4)')


