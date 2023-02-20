from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
#     """Temporal elapsed time transformer."""

#     def __init__(self, column_name):
#         self.column_name = column_name

#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         # we need this step to fit the sklearn pipeline
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:

#         # so that we do not over-write the original dataframe
#         X = X.copy()

#         for feature in self.variables:
#             X[feature] = X[self.reference_variable] - X[feature]

#         return X


# class Mapper(BaseEstimator, TransformerMixin):
#     """Categorical variable mapper."""

#     def __init__(self, variables: List[str], mappings: dict):

#         if not isinstance(variables, list):
#             raise ValueError("variables should be a list")

#         self.variables = variables
#         self.mappings = mappings

#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         # we need the fit statement to accomodate the sklearn pipeline
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         for feature in self.variables:
#             X[feature] = X[feature].map(self.mappings)

#         return X

class ExtractLetterTransformer():
    def __init__(self,column):
        self.column = column
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for i in self.column:
            X_copy[i] = X_copy[i].str.extract('([A-Za-z]+)', expand=False)
        return X_copy