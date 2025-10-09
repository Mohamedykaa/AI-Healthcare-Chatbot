import re
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Custom text preprocessing class used during model training.
    Ensures compatibility when loading the saved model.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned = [re.sub(r'[^a-zA-Z\s]', '', str(text)).lower().strip() for text in X]
        return cleaned
