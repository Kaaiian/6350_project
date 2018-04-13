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
import seaborn as sns
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

from scipy.stats import norm
from scipy import stats
#
#sns.distplot(list(mp_df[B_voight_ruess_hill].values), fit=norm)
#
#fig = plt.figure()
#res = stats.probplot(list(mp_df[B_voight_ruess_hill].values), plot=plt)

# %%
## Check to see if the y data needs to be transformed to a mora normal
## distribution
#mp_df[B_voight_ruess_hill+'log'] = np.sqrt(np.array(list(mp_df[B_voight_ruess_hill].values)))
#sns.distplot(list(mp_df[B_voight_ruess_hill+'log'].values), fit=norm)
#
#fig = plt.figure()
#res = stats.probplot(list(mp_df[B_voight_ruess_hill+'log'].values), plot=plt)
#
#series_target = np.sqrt(np.array(list(mp_df[B_voight_ruess_hill].values)))

# %%

# make list of features from formulae
features = []
targets = []
for formula, target in zip(series_formula, series_target):
    features.append(get_features(formula))
    targets.append(target)
# define the features and target for machine learning
df_features = pd.DataFrame(features)
df_targets = pd.DataFrame(targets)

# drop elements that aren't included in the elmenetal properties list. These
# will be returned as feature rows completely full of Nan values.
df_features.dropna(inplace=True, how='all')
df_targets = df_targets.loc[df_features.index]

df_features.reset_index(drop=True, inplace=True)
df_targets.reset_index(drop=True, inplace=True)
# The missing or Nan values need to be replaced before ML process. Here I have
# chosen to replace the missing values with the mean values in the data
cols = df_features.columns.values
mean_values = df_features[cols].mean()
df_features[cols]=df_features[cols].fillna(mean_values.iloc[0])


# %%
# Here we perform grouped-cross validation. Feature scaling wasn't used for
# training the model. 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
kf = KFold(n_splits=2, random_state=15, shuffle=True)

# this loop creates 'n_splits=X' number of splits. We are only keeping the
# final train and test indices of the loop, but all 5 splits could potentially 
# be tested resulting in CV test results for all data.

actual_data = []
predicted_data = []
y_test_list_nest = []
predicted_test_list_nest= []
max_value = 0
N = 0
sum_test_rmse = 0
sum_test_score = 0
sum_spearman = 0
sum_pearson = 0
for train_index, test_index in kf.split(df_features):
#    print("TRAIN:", train_index, "TEST:", test_index)
    N += 1
    X_train = df_features.reindex(train_index)
    X_test = df_features.loc[test_index]
    y_train= df_targets.loc[train_index]
    y_test = df_targets.loc[test_index]

    # convert the y-pd.DataFrames into np.arrays. This is the accepted data format 
    # for fitting the rf algorithm
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # The data is is scaled to have zero mean and unit variance. This is because
    # many algorithms in the SKLEARN package behave poorly with 'wild' features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train),
                                  index=X_train.index.values,
                                  columns=X_train.columns.values)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 index=X_test.index.values,
                                 columns=X_test.columns.values)

    # =============================================================================
    # We do random forest machine learning here. 
    # =============================================================================
    
    # The random forest regressor is called here. This uses the same parameters of 
    # the submitted publication for ML prediction of inorganic cp.
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=200,
                               max_depth=20,
                               oob_score=True,
                               random_state=15,
                               n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    # rf is now contains the trained model. rf.predict is used to generate
    # predictions. the code below uses those predictions and prints the errors.
    # This error matches publication error.
    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.stats import spearmanr, pearsonr
    
    predicted_train = rf.predict(X_train_scaled)
    predicted_test = rf.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, predicted_test))
    test_score = r2_score(y_test, predicted_test)
    spearman = spearmanr(y_test, predicted_test)
    pearson = pearsonr(y_test, predicted_test)
    
    sum_test_rmse += test_rmse
    sum_test_score += test_score
    sum_spearman += spearman[0]
    sum_pearson += pearson[0]
    
    print(f'Mean-squared-error for the test data is: {test_rmse:.3}')
    print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    print(f'Test data R-2 score: {test_score:>6.7}')
    print(f'Test data Spearman correlation: {spearman[0]:.3}')
    print(f'Test data Pearson correlation: {pearson[0]:.3}')

    # We quickly plot the actual vs predicted values. This allows us to check if 
    # the model has behaved as we expected it to. Upon inspection, we can now 
    # pickle the model confident that we are predicting the correct values.
    
    y_test_list_nest.append(y_test)
    predicted_test_list_nest.append(predicted_test)
    if y_test.max() > max_value:
        max_value = max([y_test.max(), predicted_test.max()])
    actual_data += list(y_test)
    predicted_data += list(predicted_test)


    #plt.savefig('results.png')

# %%

plt.figure(1, figsize=(8, 8))
font = {'family' : 'DejaVu Sans',
    'weight' : 'normal',
    'size'   : 18}
plt.rc('font', **font)

for y_test, predicted_test in zip(y_test_list_nest, predicted_test_list_nest):
    #    print(X_train, X_test, y_train, y_test)
    plt.plot(y_test, predicted_test, 'ro', markerfacecolor='none')
    plt.plot([0,1000],[0,1000],'k-')

max_value = 40
plt.xlabel('Actual Bulk Modulus (GPa)', fontsize=22)
plt.ylabel('Predicted Bulk Modulus (GPa)', fontsize=22)
plt.xlim((0, max_value))
plt.ylim((0, max_value))
ticks = np.linspace(0, 20, 5)
plt.xticks(ticks)
plt.yticks(ticks)
plt.legend(['Density (MPDB)','Ideal Performance'], loc='best')

plt.show()

avg_test_rmse = sum_test_rmse/N
avg_test_score = sum_test_score/N
avg_spearman = sum_spearman/N
avg_pearson = sum_pearson/N

print(f'Mean-squared-error for the test data is: {avg_test_rmse:.3}')
print(f'Test data R-2 score: {avg_test_score:>6.7}')
print(f'Test data Spearman correlation: {avg_spearman:.3}')
print(f'Test data Pearson correlation: {avg_pearson:.3}')


# %%

# Make a model to predict bulk modulus which can be used as a feature in the 
# next part where we predict conductivity 
from sklearn.ensemble import RandomForestRegressor
rf_ML_features = df_features
rf_ML_targets = df_targets.values.ravel()
rf_ML = RandomForestRegressor(n_estimators=200,
                           max_depth=20,
                           oob_score=True,
                           random_state=15,
                           n_jobs=-1)
rf_ML.fit(rf_ML_features, rf_ML_targets)

# %%
                        ##GRID SEARCH
                        #from sklearn.model_selection import GridSearchCV
                        #
                        #n_estimator_list = [200]
                        #max_depth_list = [5, 10, 20, 30, 40, 70, 150]
                        #
                        #param_grid = {'n_estimators': n_estimator_list, 'max_depth':max_depth_list}
                        #grid = GridSearchCV(estimator=rf, param_grid=param_grid)
                        #grid.fit(X_train_reduced, y_train)
                        #print(grid.best_score_)
                        #print(grid.best_estimator_.n_estimators)
                        #print(grid.best_estimator_.max_depth)
                        #
                        #

# %%
# =============================================================================
# early prepossesing. Want to look at data and make sure we have 
# everthing we expect
# =============================================================================

# load up the data for use with training the model
full_material_data = pd.read_csv('ml_data_thermal_conductivity.csv')

material_data = full_material_data.copy()

for i in material_data.index.values:
    material_data.at[i, 'ICSD'] = material_data['ICSD'].loc[i].split('/')[0]

# %%
# =============================================================================
# Main preprocessing. Here we remove duplicates, fill missing values, assign
# cross-validation splits, and scale the data.
# =============================================================================

# chose to drop duplicates or take the mean value of the duplicates
# get mean value for duplicates
material_data = material_data.groupby('ICSD').mean().reset_index()
# drop duplicates
material_data.drop_duplicates(subset=['ICSD'], keep=False, inplace=True)

# create the feature, target, and CV-grouping dataframes.
series_formula = material_data['ICSD']
series_target = material_data['Density']


# %%
# Check to see if the data can be balanced
#
#sns.distplot(np.log(df_targets), fit=norm)
#
#fig = plt.figure()
#res = stats.probplot(list(np.log(df_targets.values.transpose()[0])), plot=plt)
#
#series_target = np.array(list(np.log(df_targets.values.transpose()[0])))

# %%
# make list of features from formulae
features = []
targets = []
for formula, target in zip(series_formula, series_target):
    features.append(get_features(formula))
    targets.append(target)

# %%
# define the features and target for machine learning
df_features = pd.DataFrame(features)
df_targets = pd.DataFrame(targets)

# The missing or Nan values need to be replaced before ML process. Here I have
# chosen to replace the missing values with the mean values in the data
cols = df_features.columns.values
mean_values = df_features[cols].mean()
df_features[cols]=df_features[cols].fillna(mean_values.iloc[0])
ML_feature = rf_ML.predict(df_features)

df_features['Calculated Eg'] = ML_feature
#df_features.drop(['Calculated Eg'], axis=1, inplace=True)

#plt.figure(1, figsize=(10,10))
#plt.plot(B_feature, material_data['B'], 'ro')
#plt.plot([0,1000],[0,1000],'k-');
#plt.xlabel('Actual')
#plt.ylabel('Predicted')
#plt.xlim((0, material_data['B'].max()))
#plt.ylim((0, material_data['B'].max()))
#plt.legend(['Output','Ideal'],loc='best')
##plt.savefig('results.png')
#plt.show()

# %%
            ##GRID SEARCH
            #from sklearn.model_selection import GridSearchCV
            #
            #n_estimator_list = [200, 400]
            #max_depth_list = [10, 20, 40, 60, 100]
            #min_samples_leaf_list = [1, 10, 50, 150]
            #
            #model = RandomForestRegressor(max_features=0.1,
            #                              n_jobs=-1,
            #                              oob_score=True,
            #                              random_state=0,
#                                           n_jobs=-1)
            #
            #
            #param_grid = {'n_estimators': n_estimator_list, 'max_depth':max_depth_list, 'min_samples_leaf': min_samples_leaf_list,}
            #grid = GridSearchCV(estimator=model, param_grid=param_grid)
            #grid.fit(df_features, np.array(df_targets).ravel())
            #print(grid.best_score_)
            #print(grid.best_estimator_.n_estimators)
            #print(grid.best_estimator_.max_depth)
            #print(grid.best_estimator_.min_samples_leaf)

# %%
# Here we perform grouped-cross validation. Feature scaling wasn't used for
# training the model. 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=13, shuffle=True)

# this loop creates 'n_splits=X' number of splits. We are only keeping the
# final train and test indices of the loop, but all 5 splits could potentially 
# be tested resulting in CV test results for all data.

data_actual_kappa = []
data_prediction_kappa = []

from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
N = 0
sum_test_rmse = 0
sum_test_score = 0
sum_spearman = 0
sum_pearson = 0
y_test_list = []
predicted_test_list= []
#nested_y = []
#nested_test= []
for train_index, test_index in kf.split(df_features):
    N += 1

#    print("TRAIN:", train_index, "TEST:", test_index)
    X_train = df_features.loc[train_index]
    X_test = df_features.loc[test_index]
    y_train= df_targets.loc[train_index]
    y_test = df_targets.loc[test_index]
#    print(X_train, X_test, y_train, y_test)
    
    # convert the y-pd.DataFrames into np.arrays. This is the accepted data format 
    # for fitting the rf algorithm
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    # The data is is scaled to have zero mean and unit variance. This is because
    # many algorithms in the SKLEARN package behave poorly with 'wild' features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train),
                                  index=X_train.index.values,
                                  columns=X_train.columns.values)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 index=X_test.index.values,
                                 columns=X_test.columns.values)
    
    # Check to see if the y data needs to be transformed to a more normal
    # distributio
    # =============================================================================
    # We do random forest machine learning here. 
    # =============================================================================
    
    # The random forest regressor is called here. This uses the same parameters of 
    # the submitted publication for ML prediction of inorganic cp.
    rf = RandomForestRegressor(n_estimators=200,
                               max_depth=20,
                               oob_score=True,
                               random_state=0,
                               n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    # rf is now contains the trained model. rf.predict is used to generate
    # predictions. the code below uses those predictions and prints the errors.
    # This error matches publication error.
    
    predicted_train = rf.predict(X_train_scaled)
    predicted_test = rf.predict(X_test_scaled)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, predicted_test))
    test_score = r2_score(y_test, predicted_test)
    spearman = spearmanr(y_test, predicted_test)
    pearson = pearsonr(y_test, predicted_test)
    
    sum_test_rmse += test_rmse
    sum_test_score += test_score
    sum_spearman += spearman[0]
    sum_pearson += pearson[0]
    
    y_test_list.append(y_test)
    predicted_test_list.append(predicted_test)
    
#    nested_y.append(y_test)
#    nested_test.append(predicted_test)
    
    print(f'Mean-squared-error for the test data is: {test_rmse:.3}')
    print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    print(f'Test data R-2 score: {test_score:>6.7}')
    print(f'Test data Spearman correlation: {spearman[0]:.3}')
    print(f'Test data Pearson correlation: {pearson[0]:.3}')
    

    data_actual_kappa += list(y_test)
    data_prediction_kappa += list()
    max_value = 0
    if y_test.max() > max_value:
        max_value = y_test.max()

    #plt.savefig('results.png')

# %%

plt.figure(2, figsize=(8, 8))
font = {'family' : 'DejaVu Sans',
    'weight' : 'normal',
    'size'   : 18}
plt.rc('font', **font)
for y_test, predicted_test, n_y_test, n_predicted_test in zip(y_test_list, predicted_test_list, nested_y, nested_test):
    #    print(X_train, X_test, y_train, y_test)
    plt.plot(y_test, predicted_test, 'bo', markerfacecolor='none')
    plt.plot(n_y_test, n_predicted_test, 'rx', markerfacecolor='none')
    plt.plot([0,1000],[0,1000],'k-')

print(max_value)
max_value = 14
target_name = 'Density'
target_unit = '(g/cc)'
plt.xlabel('Actual ' + target_name + ' ' + target_unit, fontsize=22)
plt.ylabel('Predicted ' + target_name + ' ' + target_unit, fontsize=22)
plt.xlim((1, max_value))
plt.ylim((1, max_value))
ticks = np.linspace(1, 13, 7)
plt.xticks(ticks)
plt.yticks(ticks)
plt.legend([target_name + ' un-nested', target_name + ' nested', 'Ideal Performance'], loc='best')

plt.show()

avg_test_rmse = sum_test_rmse/N
avg_test_score = sum_test_score/N
avg_spearman = sum_spearman/N
avg_pearson = sum_pearson/N

print(f'Mean-squared-error for the test data is: {avg_test_rmse:.3}')
print(f'Test data R-2 score: {avg_test_score:>6.7}')
print(f'Test data Spearman correlation: {avg_spearman:.3}')
print(f'Test data Pearson correlation: {avg_pearson:.3}')



# %%

nested_y, nested_test = y_test_list, predicted_test_list

# this is an example of what the trainind data looks like when re-predicted on
# the model. Not really worth anything, but kind of interesting.
plt.figure(2, figsize=(10,10))
plt.plot(y_train, predicted_train, 'bx')
plt.plot([0,1000],[0,1000],'k-');
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.xlim((0, max_value))
plt.ylim((0, max_value))
plt.legend(['Output','Ideal'],loc='best')
#plt.savefig('results.png')

# %%

# We now look at the important features
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
importance_dic = {}
for f in range(25):
    importance_dic[X_train_scaled.columns.values[indices[f]]]=importances[indices[f]]
    print(("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])),':', X_train_scaled.columns.values[indices[f]])

# %%

# reduce the number of features to indlucde only the N most relevant based on 
# above results
N = 25
X_train_reduced = X_train_scaled.iloc[:, indices[0:N]]
X_test_reduced = X_test_scaled.iloc[:, indices[0:N]]



