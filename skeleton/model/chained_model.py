import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from model.base import BaseModel
import random

# Set seeds
seed = 33
random.seed(seed)
np.random.seed(seed)

# Chained Multi Output Classification
# The one instance manages the 3 classifiers:
#   mdl_y2 -> Type 2
#   mdl_y23 -> Type 2 + Type 3
#   mdl_y234 -> Type 2 + Type 3 + Type 4
# Any sklearn classifier can be passed to the classifier param
# (defaults is RandomForest)
class ChainedModel(BaseModel):

    # Initialize the 3 classifiers
    # 'model_name' is a written label for printed output
    # 'classifier' is the sklearn classifier passed
    def __init__(self, model_name: str, classifier=None) -> None:
        
        super(ChainedModel, self).__init__()

        self.model_name = model_name

        # Default to RandomForest if no classifier provided
        if classifier is None:
            classifier = RandomForestClassifier(
                n_estimators=1000,
                random_state=seed,
                class_weight="balanced_subsample"
            )

        # One independent instances per chain level
        # clone() copies the classifier's hyperparameters
        from sklearn.base import clone
        self.mdl_y2 = clone(classifier)
        self.mdl_y23 = clone(classifier)
        self.mdl_y234 = clone(classifier)

        # Predictions stored on self after predict() is called
        self.predictions = None  # Type 2
        self.predictions_y23 = None  # Type 2 + Type 3
        self.predictions_y234 = None  # Type 2 + Type 3 + Type 4

        self.data_transform()

    # Hook for model transformations before training
    # Not needed for RandomForest at this stage
    def data_transform(self) -> None:
        pass

    # Train the 3 classifiers from data object
    # Each receies the dame X_train, but different Y_trains
    # Data object gotten from data_model.py, which contains...
    # X_train, y_train, y23_train, y234_train 
    def train(self, data) -> None:
        print(f"\n[{self.model_name}] Training mld_y2...")
        self.mdl_y2.fit(data.X_train, data.y_train)

        print(f"[{self.model_name}] Training mdl_y23...")
        self.mdl_y23.fit(data.X_train, data.y23_train)

        print(f"[{self.model_name}] Training mdl_y234...")
        self.mdl_y234.fit(data.X_train, data.y234_train)

        print(f"[{self.model_name}] All models trained!\n")

    # Get predictions for all 3 chain levels
    def predict(self, data) -> None:
        self.predictions = self.mdl_y2.predict(data.X_test)
        self.predictions_y23 = self.mdl_y23.predict(data.X_test)
        self.predictions_y234 = self.mdl_y234.predict(data.X_test)

    # Print accuracy and full classification report for each chains
    def print_results(self, data) -> None:

        acc_y2   = accuracy_score(data.y_test,    self.predictions)
        acc_y23  = accuracy_score(data.y23_test,  self.predictions_y23)
        acc_y234 = accuracy_score(data.y234_test, self.predictions_y234)

        # Summary table
        print("\n" + "#" * 50)
        print(f"{self.model_name} — CHAINED ACCURACY SUMMARY")
        print("-" * 50)
        print(f"    1. Type 2 only : {acc_y2:.4f}  ({acc_y2 * 100:.2f}%)")
        print(f"    2. Type 2 + Type 3 : {acc_y23:.4f}  ({acc_y23 * 100:.2f}%)")
        print(f"    3. Type 2 + Type 3 + Type 4 : {acc_y234:.4f}  ({acc_y234 * 100:.2f}%)")
        print("#" * 50)

        # Full classifications report
        print("\nClassification Report : (Type 2)")
        print(classification_report(
            data.y_test, self.predictions, zero_division=0))

        print("\nClassification Report : (Type 2 + Type 3)")
        print(classification_report(
            data.y23_test, self.predictions_y23, zero_division=0))

        print("\nClassification Report : (Type 2 + Type 3 + Type 4)")
        print(classification_report(
            data.y234_test, self.predictions_y234, zero_division=0))