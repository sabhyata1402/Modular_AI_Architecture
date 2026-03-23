# PURPOSE:
#   Defines the Data class — encapsulates ALL training and testing arrays
#   into one single object passed to every ML model.
#
# ARCHITECTURAL PRINCIPLE:
#   Encapsulation (Feature 2).
#   CA brief (B): "encapsulate all your required input data in one object and pass that object as data elements to all ML models."
#   CA brief (B): "remove the records for the classes having very few instances"
#
# CA1 EXTENSION:
#   The base Data class stores X_train, X_test, y_train, y_test.
#   Extended for Chained Multi-Output (Design Choice 1):
#       y2_3_train / y23_test   — "Type 2_Type 3" combined label
#       y2_3_4_train / y2_3_4_test — "Type 2_Type 3_Type 4" combined label
#
# NaN HANDLING:
#   Type 3 and Type 4 have NaN values in the real data (35+ rows).
#   Rows with NaN in Type 3 or Type 4 are filtered BEFORE building
#   chained labels — prevents invalid "Suggestion_nan" labels.
#   Rows with only Type 2 filled are still used by the base Data class.
#
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

# Encapsulates all training and testing data in a single object. Built once per group in main.py, then passed as the single argument to model.train(data), model.predict(data), model.print_results(data).
class Data():
    """
    Attributes (base — from lecturer's solution):
        X_train, X_test   : TF-IDF feature matrix splits
        y_train, y_test   : Type 2 label splits
        y                 : all Type 2 labels (before split)
        classes           : valid class values (those with >= 3 records)
        embeddings        : full X matrix before splitting

    Attributes (CA1 extension — chained labels):
        y2_3_train, y23_test   : "Type 2_Type 3" combined label splits
        y2_3_4_train, y2_3_4_test : "Type 2_Type 3_Type 4" combined label splits
    """

    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
    

        # ── Step 1: Extract primary label (Type 2) ────────────────────────────
        # Config.CLASS_COL = 'Type 2'
        df = df.reset_index(drop=True)
        y = df[Config.CLASS_COL].to_numpy()
        y_series = pd.Series(y)

        # CA brief (B): "remove the records for the classes having very few instances"
        # Classes with fewer than 3 records cannot be reliably split into train/test sets and provide too little data to learn from.
        good_y_value = y_series.value_counts()[
            y_series.value_counts() >= 3
        ].index

        if len(good_y_value) < 1:
            print("None of the class have more than 3 records: Skipping ...")
            self.X_train = None
            return

        # Filter X and y to only rows with valid (non-rare) Type 2 classes
        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]
        df_good = df[y_series.isin(good_y_value)].copy().reset_index(drop=True)

        # ── Step 2: Build chained labels  ──────────────────────
        # Filter further: only rows where Type 3 AND Type 4 are both non-null.
        valid_mask = df_good[['Type 3', 'Type 4']].notna().all(axis=1)
        df_chain   = df_good[valid_mask].copy().reset_index(drop=True)
        X_chain    = X_good[valid_mask.values]
        y_chain    = y_good[valid_mask.values]

        # Level 2: "Type 2_Type 3"
        # Real examples: 'Suggestion_Payment', 'Problem/Fault_AppGallery-Install/Upgrade'
        y2_3_arr = (
            df_chain['Type 2'].astype(str) + "_" +
            df_chain['Type 3'].astype(str)
        ).to_numpy()

        # Level 3: "Type 2_Type 3_Type 4"
        # Real examples: 'Suggestion_Payment_Subscription cancellation'
        y2_3_4_arr = (
            df_chain['Type 2'].astype(str) + "_" +
            df_chain['Type 3'].astype(str) + "_" +
            df_chain['Type 4'].astype(str)
        ).to_numpy()

        # ── Step 3: Adjusted test size ────────────────────────────────────────
        # Keep ~20% of the original group as test data.
        new_test_size = X.shape[0] * 0.2 / X_chain.shape[0]

        # ── Step 4: Single train/test split for ALL arrays ────────────────────
        (self.X_train,    self.X_test,
         self.y_train,    self.y_test,
         self.y2_3_train,  self.y2_3_test,
         self.y2_3_4_train, self.y2_3_4_test) = train_test_split(
            X_chain, y_chain, y2_3_arr, y2_3_4_arr,
            test_size=new_test_size,
            random_state=0,
            stratify=y_chain
        )

        self.y          = y_chain
        self.classes    = good_y_value
        self.embeddings = X   # full matrix before filtering (used by model __init__)


    def get_type(self):
        """Return all Type 2 labels (combined before split)."""
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_embeddings(self):
        """Return full TF-IDF matrix (before train/test split)."""
        return self.embeddings
