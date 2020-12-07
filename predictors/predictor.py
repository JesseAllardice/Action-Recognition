"""
ABC for a predictor.
similar implemenation to sk-learn estimators
"""

from abc import ABC, abstractmethod

class Predictor(ABC):
    @abstractmethod
    def predict(self): pass

    @abstractmethod
    def transform(self): pass

    @abstractmethod
    def fit(self): pass

    @abstractmethod
    def fit_transform(self): pass