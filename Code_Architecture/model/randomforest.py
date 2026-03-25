# Single label RF classifier for Type 2
# Seves as baseline model
# 
# Inherits from BaseModel, implements train/predict/print_results

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from model.base import BaseModel
import random

# Seeds
seed = 0
random.seed(seed)
np.random.seed(seed)

# Baseline classifier for Type 2
# Used as the baseline to compare against ChainedModel results
class RandomForest(BaseModel):

    # Initialise RF classifier
    def __init__(self, model_name: str) -> None:
    
        super(RandomForest, self).__init__()

        self.model_name  = model_name
        self.predictions = None

        # n_estimators - 1000 decision trees for stable predictions
        # class_weight='balanced_subsample' - handles class imbalance
        self.mdl = RandomForestClassifier(
            n_estimators=1000,
            random_state=seed,
            class_weight='balanced_subsample'
        )

        self.data_transform()

    # Hook for data transformations before training
     # Not needed for RandomForest
    def data_transform(self) -> None:
        pass

    # Train classifier on Type 2 labels
    def train(self, data) -> None:
        print(f"\n[{self.model_name}] Training on Type 2 labels...")
        self.mdl.fit(data.X_train, data.y_train)
        print(f"[{self.model_name}] Training is compete")

    # Generate predictions on Type 2 labels
    def predict(self, data) -> None:
        self.predictions = self.mdl.predict(data.X_test)

    # Print classification report
    def print_results(self, data) -> None:
        print(f"\n{'='*60}")
        print(f" {self.model_name} — Baseline Classifier (Type 2 only)")
        print(f"{'='*60}")
        print(classification_report(
            data.y_test,
            self.predictions,
            zero_division=0
        ))