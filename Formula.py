import pandas as pd
import pymatgen as mg
import numpy as np

class Formula:

    def __init__(self, formula, element_data):
        self.feature_vector = None
        self.formula = formula
        self.get_features(formula, element_data)

    def get_feature_vector(self):
        return self.feature_vector

    def get_features(self, formula, element_data):
        '''
        Input
        ----------
        formula: string
            put a valid chemical fomula as a sting. Example( 'NaCl')

        Output
        ----------
        features: np.array()
            This is an 1x252 length array containing feature values for use in the
            machine learning model.
        '''
        try:
            fractional_composition = mg.Composition(formula).fractional_composition.as_dict()
            element_composition = mg.Composition(formula).element_composition.as_dict()
            avg_feature = np.zeros(len(element_data.iloc[0]))
            sum_feature = np.zeros(len(element_data.iloc[0]))
            for key in fractional_composition:
                try:
                    avg_feature += element_data.loc[key].values * fractional_composition[key]
                    sum_feature += element_data.loc[key].values * element_composition[key]
                except:
                    print('The element:', key, 'is not currently supported in our database')
                    self.feature_vector = np.array([np.nan] * len(element_data.iloc[0]) * 4)
            var_feature = element_data.loc[list(fractional_composition.keys())].var()
            range_feature = element_data.loc[list(fractional_composition.keys())].max() - element_data.loc[
                list(fractional_composition.keys())].min()

            features = pd.DataFrame(
                np.concatenate([avg_feature, sum_feature, np.array(var_feature), np.array(range_feature)]))
            features = np.concatenate([avg_feature, sum_feature, np.array(var_feature), np.array(range_feature)])
            self.feature_vector = features.transpose()
        except:
            print(
            'There was and error with the Formula: ' + formula + ', this is a general exception with an unkown error')
            self.feature_vector = [np.nan] * len(element_data.iloc[0]) * 4
