from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np

log_list = [
    #'Elastic Anisotropy',
    'Energy Above Convex Hull',
    'Enthalpy of Formation',
    'Piezoelectric Modulus',
    'Poisson\'s Ratio',
    'Unit Cell Volume'
]

def crossValidate(df_features, df_targets, calc_property):
    N = 5
    kf = KFold(n_splits=N, random_state=15, shuffle=True)
    actual_data = []
    predicted_data = []
    y_test_list_nest = []
    predicted_test_list_nest = []
    sum_percent = 0
    max_value = 0
    sum_test_rmse = 0
    sum_test_score = 0
    sum_spearman = 0
    sum_pearson = 0
    metrics_string = ''

    if calc_property in log_list:
        df_targets = np.log(df_targets)


    for train_index, test_index in kf.split(df_features):
        X_train = df_features.reindex(train_index)
        X_test = df_features.loc[test_index]
        y_train = df_targets.loc[train_index]
        y_test = df_targets.loc[test_index]

        # convert the y-pd.DataFrames into np.arrays. This is the accepted data format
        # for fitting the rf algorithm
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

        # The data is is scaled to have zero mean and unit variance. This is because
        # many algorithms in the SKLEARN package behave poorly with 'wild' features
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train),
                                      index=X_train.index.values,
                                      columns=X_train.columns.values)
        X_test = pd.DataFrame(scaler.transform(X_test),
                                     index=X_test.index.values,
                                     columns=X_test.columns.values)

        normalizer = Normalizer().fit(X_train)
        X_train = normalizer.transform(X_train)
        normalizer = Normalizer().fit(X_test)
        X_test = normalizer.transform(X_test)

        # =============================================================================
        # We do random forest machine learning here.
        # =============================================================================

        # The random forest regressor is called here. This uses the same parameters of
        # the submitted publication for ML prediction of inorganic cp.

        rf = RandomForestRegressor(n_estimators=200,
                                   max_depth=20,
                                   oob_score=True,
                                   random_state=15,
                                   n_jobs=-1)
        rf.fit(X_train, y_train)

        # rf is now contains the trained model. rf.predict is used to generate
        # predictions. the code below uses those predictions and prints the errors.
        # This error matches publication error.


        predicted_train = rf.predict(X_train)
        predicted_test = rf.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, predicted_test))
        test_score = r2_score(y_test, predicted_test)
        spearman = spearmanr(y_test, predicted_test)
        pearson = pearsonr(y_test, predicted_test)
        avg_percent_error = np.mean(np.abs(np.array(y_test)-np.array(predicted_test))/(abs(np.array(y_test))+.00000000000001)*100)
    
        sum_test_rmse += test_rmse
        sum_test_score += test_score
        sum_percent += avg_percent_error
        sum_spearman += spearman[0]
        sum_pearson += pearson[0]

        metrics_string += ('___________________________________________\n'
                         'Mean-squared-error for the test data is: ' + str(test_rmse) + '\n'
                        'Test data R-2 score: ' + str(test_score) + '\n'
                        'Average percent error: ' + str(avg_percent_error) + '\n'
                        '___________________________________________\n')

        print('validation finished')
#        print(f'Root-mean-squared-error for the test data is: {test_rmse:.3}')
#        print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
#        print(f'Test data R-2 score: {test_score:>6.7}')
#        print(f'Test data Spearman correlation: {spearman[0]:.3}')
#        print(f'Test data Pearson correlation: {pearson[0]:.3}')

        # We quickly plot the actual vs predicted values. This allows us to check if
        # the model has behaved as we expected it to. Upon inspection, we can now
        # pickle the model confident that we are predicting the correct values.

        y_test_list_nest.append(y_test)
        predicted_test_list_nest.append(predicted_test)
        if y_test.max() > max_value:
            max_value = max([y_test.max(), predicted_test.max()])
        actual_data += list(y_test)
        predicted_data += list(predicted_test)


    print("-----------------------------")
    avg_test_rmse = sum_test_rmse / N
    avg_test_score = sum_test_score / N
    avg_spearman = sum_spearman / N
    avg_pearson = sum_pearson / N
    avg_percent = sum_percent / N
    
    metrics_string += ("--------------combined metrics---------------\n"
                       'Root-mean-squared-error for the test data is: ' + str(avg_test_rmse)  +'\n'
                       'Test data R-2 score: ' + str(avg_test_score) +'\n'
                       'Average percent error: ' + str(avg_percent) +'\n'
                       )
    print(metrics_string)
#    print(f'Mean-squared-error for the test data is: {avg_test_rmse:.3}')
#    print(f'Test data R-2 score: {avg_test_score:>6.7}')
#    print(f'Test data Spearman correlation: {avg_spearman:.3}')
#    print(f'Test data Pearson correlation: {avg_pearson:.3}')
    
    return (y_test_list_nest, predicted_test_list_nest, metrics_string)

def createModel(df_features, df_targets):
    rf_ML_features = df_features
    rf_ML_targets = df_targets.values.ravel()
    rf_ML = RandomForestRegressor(n_estimators=200,
                                  max_depth=20,
                                  oob_score=True,
                                  random_state=15,
                                  n_jobs=-1)
    return rf_ML.fit(rf_ML_features, rf_ML_targets)