from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

# Abstract Base Class (ABC) that defines interfaces fro all ML models
# New models that are added have to inherit from BaseModel and implement
# the abstractmethods
class BaseModel(ABC):

    # This is the base constructor
    # Subclasses run the super().__ini__() method to define model-
    # specific values
    def __init__(self) -> None:
        pass
    
    # This trains the model with the provided data object
    # The data param is a container object that holds the X_train and 
    # Y_train values
    @abstractmethod
    def train(self, data) -> None:
        pass

    # This generates predictions with the trained model
    # The data param holds the X_test
    # Predictions should be stored on self.predictions to access them
    # with print_results later
    @abstractmethod
    def predict(self, data) -> None:
        pass

    # This prints the evaluations for the model predictions
    # The data param holds the Y_test values
    # This method compares predictions against Y_test and prints a 
    # classification report or the like
    @abstractmethod
    def print_results(self, data) -> None:
        pass

    # This applies any data transformations for different models
    # Eg. for reshaping or encoding inputs
    @abstractmethod
    def data_transform(self) -> None:
        pass

    # This is a method to build models with hyperparameters
    # It's not abstract to allow models with default values
    # The values param is a dictionary for hyperparameter names : values
    # Eg. model.build({'n_estimators': 100})
    # Self is returned to allow method chaining
    def build(self, values: dict = {}) -> 'BaseModel':
        # Only update if values is a valid dictionary
        if isinstance(values, dict):
            self.__dict__.update(values)
        return self