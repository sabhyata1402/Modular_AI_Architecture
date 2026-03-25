# To define Data class, and encapsulate all training and testing
#
# Encapsulation : All data lives in the one object. Models do not
#   reach outside the object for the data. New label levels added
#   only requires updating this class
# 
# Base attributes (X_train, X_test, y_train, y_test)
# Extended with chained label arrays...
#   y23_train / y23_test for Type2_Type3
#   y234_train / y234_test for Type2_Type3_Type4
# 
# NaN handling:
#   Type 3 and Type 4 have NaN values in the real data
#   Rows with NaN in Type 3 or Type 4 are filtered before building
#   chained labels

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

# Encapsulates all training and testing data in single object
#
# Built once per group in main.py, then passed as argument to
# model.train(data), and model.predict(data), and model.print_results(data)
class Data:

    # Build the train/test splits from the TF-IDF matrix and df
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:

        # Extract and validate Type 2 labels
        # Reset index to ensure the alignment between the X and df
        df = df.reset_index(drop=True)
        y = df[Config.CLASS_COL].to_numpy()
        y_series = pd.Series(y)

        # remove classes with fewer than 3 records
        # Too few records cannot be reliably split into train/test
        # and provide too little signal for the model to learn from
        good_y_value = y_series.value_counts()[
            y_series.value_counts() >= 3
        ].index

        if len(good_y_value) < 1:
            print("No class has more than 3 records, skipping group")
            self.X_train = None
            return

        # Filter rows to only valid Type 2 classes
        mask = y_series.isin(good_y_value)
        y_good  = y[mask]
        X_good  = X[mask]
        df_good = df[mask].copy().reset_index(drop=True)

        # Filter NaNs and build chained labels
        # Only rows where Type 3 and Type 4 are not null can
        # be used for chained classification
        valid_mask = df_good[['Type 3', 'Type 4']].notna().all(axis=1)
        df_chain = df_good[valid_mask].copy().reset_index(drop=True)
        X_chain  = X_good[valid_mask.values]
        y_chain  = y_good[valid_mask.values]

        if len(y_chain) < 3:
            print("Insufficient data after NaN filtering, skipping group")
            self.X_train = None
            return

        # "Type2_Type3" combined label
        # example - "Problem/Fault_AppGallery-Install/Upgrade"
        y23_arr = (
            df_chain['Type 2'].astype(str) + "_" +
            df_chain['Type 3'].astype(str)
        ).to_numpy()

        # "Type2_Type3_Type4" combined label
        # eg - "Problem/Fault_AppGallery-Install/Upgrade_Can't update Apps"
        y234_arr = (
            df_chain['Type 2'].astype(str) + "_" +
            df_chain['Type 3'].astype(str) + "_" +
            df_chain['Type 4'].astype(str)
        ).to_numpy()

        # Calculate adjusted test size
        # Target around 20% of the orig group size as test data
        # Clamped to max 0.4 to prevent crashes on small groups
        # where NaN filtering removed many rows
        new_test_size = min(X.shape[0] * 0.2 / X_chain.shape[0], 0.4)

        #  Single split for all arrays simultaneously
        # One call ensures all arrays are split identically —
        # row i in X_train always corresponds to row i in y_train,
        # y23_train, and y234_train
        # stratify=y_chain preserves class proportions in both splits
        (self.X_train,   self.X_test,
         self.y_train,   self.y_test,
         self.y23_train, self.y23_test,
         self.y234_train, self.y234_test) = train_test_split(
            X_chain, y_chain, y23_arr, y234_arr,
            test_size=new_test_size,
            random_state=seed,
            stratify=y_chain
        )

        # Store full arrays for reference
        self.y = y_chain    # all Type 2 labels before split
        self.classes = good_y_value     # valid class names
        self.embeddings = X     # full TF-IDF matrix before filtering

    # GETTERS
    # Provide controlled access to internal data arrays

    # Full TF-IDF matrix
    def get_embeddings(self) -> np.ndarray:
        return self.embeddings

    # All Type 2 labels
    def get_type(self) -> np.ndarray:
        return self.y

    def get_X_train(self) -> np.ndarray:
        return self.X_train

    def get_X_test(self) -> np.ndarray:
        return self.X_test

    # Type 2 labels - train split
    def get_y_train(self) -> np.ndarray:
        return self.y_train

    # Type 2 labels - test split
    def get_y_test(self) -> np.ndarray:
        return self.y_test

    # Chained labels Type2_Type3 - train split
    def get_y23_train(self) -> np.ndarray:
        return self.y23_train

    # Chained labels Type2_Type3 - test split
    def get_y23_test(self) -> np.ndarray:
        return self.y23_test

    # Chained labels Type2_Type3_Type4 - train split
    def get_y234_train(self) -> np.ndarray:
        return self.y234_train

    # Chained labels Type2_Type3_Type4 - test split
    def get_y234_test(self) -> np.ndarray:
        return self.y234_test