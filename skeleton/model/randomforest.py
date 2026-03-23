import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random

# Global configurations for model evaluation and reproducibility
num_folds = 0
seed = 0

# Fix random seeds for reproducibility
np.random.seed(seed)
random.seed(seed)

# Single-label Random Forest classifier for Type 2 classification.
    # Inherits from BaseModel (Abstraction).
    # The main controller interacts with this model ONLY through the
    # three abstract methods: train(), predict(), print_results().
class RandomForest(BaseModel):
   
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        """
        Initializes the model metadata, inputs, and the Scikit-Learn estimator.
        """
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        
        # Initialize Scikit-Learn's Random Forest classifier:
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None
        self.data_transform()

    # Train the Random Forest on the Type 2 labels from the Data object.

        # Uses:
        #     data.X_train — TF-IDF feature matrix (training rows)
        #     data.y_train — Type 2 class labels for training rows

        # The fitted model is stored back on self.mdl so predict() can use it.
        
    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    # Generate Type 2 predictions on the test set from the Data object.
    def predict(self, data):
        """
        Executes model inference on the testing feature set (X_test) and caches predictions.
        """
        predictions = self.mdl.predict(data.X_test)
        self.predictions = predictions

    # Print a full classification report comparing predictions vs true labels.
    def print_results(self, data):
        """
        Evaluates the cached predictions against the ground truth labels (data.y_test).
        Outputs a comprehensive classification report (precision, recall, f1-score, support).
        """
        print(classification_report(data.y_test, self.predictions))

    # Apply any data transformations required by this model.
    def data_transform(self) -> None:
        pass

