"""
inherients from predictor
"""

#from predictors.predictor import Predictor
from predictors import predictor

class PosePredictor(predictor.Predictor):
    # Inheriteted abstract methods
    def predict(self): pass

    def transform(self): pass

    def fit(self): pass

    def fit_transform(self): pass

    # methods

def main():
    pass

if __name__ == "__main__":
    main()