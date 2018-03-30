import pandas as pd

class FeatureTargetMatrix:

    def __init__(self):
        self.features = [] # List of formula features
        self.targets = [] # List of targets, in parallel with formula feature vectors
        self.df_features = None
        self.df_targets = None

    #
    def addFormula(self, new_formula_feature_vector, new_target):
        self.features.append(new_formula_feature_vector)
        self.targets.append(new_target)

    # Called after all of the feature vectors have been collected
    def createDataFrame(self):
        self.df_features = pd.DataFrame(self.features)
        self.df_targets = pd.DataFrame(self.targets)
        self.cleanFeatureMatrix()

    def cleanFeatureMatrix(self):
        # drop elements that aren't included in the elmenetal properties list. These
        # will be returned as feature rows completely full of Nan values.
        self.df_features.dropna(inplace=True, how='all')
        self.df_targets = self.df_targets.loc[self.df_features.index]

        self.df_features.reset_index(drop=True, inplace=True)
        self.df_targets.reset_index(drop=True, inplace=True)
        # The missing or Nan values need to be replaced before ML process. Here I have
        # chosen to replace the missing values with the mean values in the data


        '''
        POTENTIAL UPDATE:
        Might use machine learning to fill in the missing data rather than averages
        '''


        cols = self.df_features.columns.values
        mean_values = self.df_features[cols].mean()
        self.df_features[cols] = self.df_features[cols].fillna(mean_values.iloc[0])

    # Getter functions
    def get_df_features(self):
        return self.df_features
    def get_feature(self):
        return self.features