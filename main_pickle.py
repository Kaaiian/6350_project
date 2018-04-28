from Utility import *
from MachineLearning import *
import copy
import pickle

df = pd.read_excel(r'band_gap-formula-property.xlsx')
df.columns = ['formula', 'target']
element_data = pd.read_csv(r'simple_element_properties.csv', index_col=0)

propery_list = [
    'Band Gap',
    #'Bulk Modulus, Voigt-Reuss-Hill',
    'Density',
    #'Elastic Anisotropy', this was removed because the dataset for this attibute performed poorly with Machine Learning
    #'Energy Above Convex Hull',
    #'Enthalpy of Formation',
    #'Piezoelectric Modulus',
    #"Poisson's Ratio",
    #'Shear Modulus, Voigt-Reuss-Hill',
    #'Total Magnetization',
    'Unit Cell Volume']

log_list = [
    'Elastic Anisotropy'
]

if __name__ == "__main__":

    print('\n\ncalculating base performance\n-------------------------------')
    # This gathers the input data and makes sure that it is suitable for machine learning
    feature_matrix_df, metrics_string = predictDataFrame(df, element_data)
    features = copy.deepcopy(feature_matrix_df.get_df_features())
    targets = copy.deepcopy(feature_matrix_df.get_df_targets())

    f = open('metrics/' + 'metrics_' + 'original_band_gap' + '.txt', 'a')
    f.write(metrics_string)
    f.close()


    def multifeature(features, targets):
        for calc_property in propery_list:

            # Create the machine learning model using the feature matrix obtained from the calculated properties
            filename = "pickles/" + calc_property + "_newProperty.sav"
            newProperty = pickle.load(open(filename, 'rb'))

            # Append this new property vector to the end of df
            new_column = 'Predicted ' + calc_property
            features[new_column] = newProperty

            f = open('metrics/' + 'metrics_' + calc_property + '.txt', 'a')
            f.write(metrics_string)
            f.close()

        return features, targets


    def iterative(features, targets):
        models = []
        #models = [pickle.load(open("pickles/iters_Band Gap_Model.sav", 'rb')),
        #          pickle.load(open("pickles/iters_Density_Model.sav", 'rb'))]
        for calc_property in propery_list:

            # Get the dataframe of calculated values corresponding to the specified target
            df_calc_prop = get_MP_formula_property(property_of_interest=calc_property)

            print('\n \n \nTesting machine learning for calculated', calc_property, '\n-------------------------------')
            # Calculate the feature matrix based on the the calculated properties
            feature_matrix_df_calc_prop = predictDataFrame_nocross(df_calc_prop, element_data)

            calc_features = feature_matrix_df_calc_prop.get_df_features()

            # create a for loop
            # that handles iterative adding of features
            for model in models:
                newProperty = model.predict(calc_features)
                # Append this new property vector to the end of df
                new_column = 'Predicted ' + calc_property
                calc_features[new_column] = newProperty

            ml_model = pickle.load(open("pickles/iters_" + calc_property + "_Model.sav", 'rb'))

            # Create the machine learning model using the feature matrix obtained from the calculated properties

            models.append(ml_model)

            # Use the model to predict new properties based on the original dataframe (df)
            newProperty = ml_model.predict(features)
            pickle.dump(newProperty, open("pickles/iters_" + calc_property + "_newProperty.sav", 'wb'))

            # Append this new property vector to the end of df
            new_column = 'Predicted ' + calc_property
            features[new_column] = newProperty

            #f = open('metrics/' + 'metrics_' + calc_property + '.txt', 'a')
            #f.write(metrics_string)
            #f.close()
        pickle.dump(features, open("pickles/iters_" + calc_property + "_finalFeatures.sav", 'wb'))
        return features, targets

    # multi_features, multi_targets = multifeature(features, targets)
    iter_features, iter_targets = iterative(features, targets)

    print('\nfinal performance with multi-feature nested features\n-------------------------------')
    # metrics_string = predict_feature_vector(multi_features, multi_targets, calc_property="multi-feature_band_gap")
    metrics_string = predict_feature_vector(iter_features, iter_targets, calc_property="multi-feature_band_gap")

    f = open('metrics/' + 'metrics_multi_feature_band_gap.txt', 'a')
    f.write(metrics_string)
    f.close()