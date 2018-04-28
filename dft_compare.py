# For every method in json pif files
    # If method is in our database
        # Find difference
        # Increment average count

# Calculate average

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:10:38 2018

@author: Kaai
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:42:55 2018

@author: Kaai
"""

  # -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:51:11 2018

@author: Kaai
"""

import numpy as np
import pandas as pd
import pickle
import pymatgen as mg
import os
import matplotlib.pyplot as plt
from sklearn import datasets


# %%
element_data = pd.read_csv(r'simple_element_properties.csv', index_col=0)

def get_features(formula):
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
                return np.array([np.nan]*len(element_data.iloc[0])*4)
        var_feature = element_data.loc[list(fractional_composition.keys())].var()
        range_feature = element_data.loc[list(fractional_composition.keys())].max()-element_data.loc[list(fractional_composition.keys())].min()

        features = pd.DataFrame(np.concatenate([avg_feature, sum_feature, np.array(var_feature), np.array(range_feature)]))
        features = np.concatenate([avg_feature, sum_feature, np.array(var_feature), np.array(range_feature)])
        return features.transpose()
    except:
        print('There was and error with the Formula: '+ formula + ', this is a general exception with an unkown error')
        return [np.nan]*len(element_data.iloc[0])*4
# %%

B_reuss = 'Bulk Modulus, Reuss'
B_voight = 'Bulk Modulus, Voigt'
B_voight_ruess_hill = 'Bulk Modulus, Voigt-Reuss-Hill'
band_gap = 'Band Gap'
density = 'Density'
#properties_of_interest = [B_reuss, B_voight]
#properties_of_interest = [B_voight_ruess_hill]
properties_of_interest = [band_gap]

mp_df_dict ={}
N = 0

jsons = []

for i in range(1,11):
    data_sheet = 'mp_all_pif-merged-' + str(i) + '.json'
    jsons.append(data_sheet)

for data_sheet in jsons:
    mp_raw_df = pd.read_json(data_sheet)
    for formula, properties in zip(mp_raw_df['chemicalFormula'], mp_raw_df['properties']):
        mp_df_dict[N] = {}
        for item in properties:
            if item['name'] in properties_of_interest:
                if item['scalars'] > 0:
                    mp_df_dict[N]['formula'] = formula
                    mp_df_dict[N][item['name']] = float(item['scalars'])
        N += 1

mp_df = pd.DataFrame(mp_df_dict).transpose()
mp_df.dropna(inplace=True)

mp_df[properties_of_interest] = mp_df[properties_of_interest].astype(float)
# chose to drop duplicates or take the mean value of the duplicates
# get mean value for duplicates here
mp_df = mp_df.groupby('formula').mean().reset_index()
#
# drop duplicates  here
mp_df.drop_duplicates(subset=['formula'], keep=False, inplace=True)


# %%
# create the feature, target, and CV-grouping dataframes.
series_formula = mp_df['formula']

# choices: B_reuss, B_voight, B_voight_ruess_hill
#series_target = mp_df[B_voight_ruess_hill]
#series_target = mp_df[band_gap]
series_target = mp_df[band_gap]


df = pd.read_excel(r'band_gap-formula-property.xlsx')
df.columns = ['formula', 'target']

list_experimental = []
list_dft = []

print(df['formula'])

for x in range(0, series_target.shape[0]):
    print(series_formula.iloc[x])
    if series_formula.iloc[x] in df['formula']:
        list_dft.append(series_target[x])
        list_experimental.append(df['target'].loc[df['formula'] == series_formula.iloc[x]])

dft_array = np.array(list_dft)
exp_array = np.array(list_experimental)

diff_array = dft_array - exp_array
average = np.average(diff_array)
print(average)