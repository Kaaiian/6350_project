import pandas as pd

class FeatureMatrix:

    def __init__(self):
        self.features = [] # List of formula features
        self.df_features = None

    #
    def addFormula(self, new_formula_feature_vector):
        self.features.append(new_formula_feature_vector)

    # Called after all of the feature vectors have been collected
    def createDataFrame(self):
        self.df_features = pd.DataFrame(self.features)

    # Getter functions
    def get_df_features(self):
        return self.df_features
    def get_feature(self):
        return self.features