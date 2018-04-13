import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from Formula import *
from FeatureTargetMatrix import *
from MachineLearning import *

def predictDataFrame(df, element_data, calc_property='experimental_band_gap'):
    feature_matrix = FeatureTargetMatrix()
    series_formula = df['formula']
    series_target = df['target']
    for formula, target in zip(series_formula, series_target):
        new_formula = Formula(formula, target, element_data)
        feature_matrix.addFormula(new_formula.get_feature_vector(), new_formula.get_target())
    feature_matrix.createDataFrame()
    #    print(feature_matrix.get_df_features())
    #    print(feature_matrix.get_df_targets())

    metrics_string = predict_feature_vector(feature_matrix.get_df_features(), feature_matrix.get_df_targets(), calc_property)

    return feature_matrix, metrics_string

def predict_feature_vector(features, targets, calc_property='experimental_band_gap'):
    y_test_list_nest, predicted_test_list_nest, metrics_string = crossValidate(features, targets, calc_property)
    plot_mlOutput(y_test_list_nest, predicted_test_list_nest, calc_property)
    return metrics_string


def plot_mlOutput(y_test_list_nest, predicted_test_list_nest, calc_property):
    plt.figure(figsize=(8, 8))
    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)

    max_value = 0
    for y_test, predicted_test in zip(y_test_list_nest, predicted_test_list_nest):
        #    print(X_train, X_test, y_train, y_test)
        plt.plot(y_test, predicted_test, 'ro', markerfacecolor='none')
        plt.plot([0, 5000], [0, 5000], 'k-')
        if max(y_test) > max_value:
            max_value = max(y_test)

    plt.xlabel('Actual '+calc_property, fontsize=22)
    plt.ylabel('Predicted '+calc_property, fontsize=22)
    plt.xlim((0, max_value))
    plt.ylim((0, max_value))
    ticks = np.linspace(0,max_value, 5)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.legend(['ML-performance', 'Ideal Performance'], loc='best')
    
    pathname = os.getcwd()
    filename = calc_property + '_pred_vs_act'
    extension = '.png'
    plt.savefig(pathname + '/figures/' + filename + extension)
#    plt.show()

def get_MP_formula_property(property_of_interest='Band Gap'):

    '''
    Input 
    ------------
    Property_of_interest: str,
        Possible values:
             'Band Gap',
             'Bulk Modulus, Reuss',
             'Bulk Modulus, Voigt',
             'Bulk Modulus, Voigt-Reuss-Hill',
             'CIF File',
             'Compliance Tensor',
             'Crystal System',
             'Density',
             'Elastic Anisotropy',
             'Energy Above Convex Hull',
             'Enthalpy of Formation',
             'Full Formula',
             'Number of Elements',
             'Oxide Type',
             'Piezoelectric Direction',
             'Piezoelectric Modulus',
             'Piezoelectric Tensor',
             'Point Group',
             "Poisson's Ratio",
             'Shear Modulus, Reuss',
             'Shear Modulus, Voigt',
             'Shear Modulus, Voigt-Reuss-Hill',
             'Space Group',
             'Space Group Number',
             'Stiffness Tensor',
             'Total Magnetization',
             'Unit Cell Volume',
             'VASP Energy for Structure'
    '''
    
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
                if item['name'] in property_of_interest:
                    if item['scalars'] > 0:
                        mp_df_dict[N]['formula'] = formula
                        mp_df_dict[N][item['name']] = float(item['scalars'])
            N += 1

    mp_df = pd.DataFrame(mp_df_dict).transpose()
    mp_df.dropna(inplace=True)
    
    mp_df['target'] = mp_df[property_of_interest].astype(float)
    # chose to drop duplicates or take the mean value of the duplicates
    # get mean value for duplicates here
    mp_df = mp_df.groupby('formula').mean().reset_index()
    #
    # drop duplicates  here
    mp_df.drop_duplicates(subset=['formula'], keep=False, inplace=True)
    
    cutoff = 10
    if len(mp_df) > cutoff:
        mp_df = mp_df.sample(n=cutoff)
    return mp_df

