from Utility import *
from MachineLearning import *

df = pd.read_excel(r'band_gap-formula-property.xlsx')
df.columns = ['formula', 'target']
element_data = pd.read_csv(r'simple_element_properties.csv', index_col=0)

propery_list = [
    'Band Gap',
    'Bulk Modulus, Voigt-Reuss-Hill',
#    'Density',
    'Elastic Anisotropy',
    'Energy Above Convex Hull',
    'Enthalpy of Formation',
    'Piezoelectric Modulus',
    "Poisson's Ratio",
    'Shear Modulus, Voigt-Reuss-Hill',
    'Total Magnetization',
    'Unit Cell Volume']

if __name__ == "__main__":

    # This gathers the input data and makes sure that it is suitable for machine learning
    feature_matrix_df = predictDataFrame(df, element_data)
    features = feature_matrix_df.get_df_features()
    targets = feature_matrix_df.get_df_targets()

    for calc_property in propery_list[0:1]:

        # Get the dataframe of calculated values corresponding to the specified target
        df_calc_prop = get_MP_formula_property(property_of_interest=calc_property)
        
        print('\nTesting machine learning for calculated', calc_property, '...')
        # Calculate the feature matrix based on the the calculated properties
        feature_matrix_df_calc_prop = predictDataFrame(df_calc_prop, element_data)
        
        # Create the machine learning model using the feature matrix obtained from the calculated properties
        ml_model = createModel(feature_matrix_df_calc_prop.get_df_features(), feature_matrix_df_calc_prop.get_df_targets())

        # Use the model to predict new properties based on the original dataframe (df)
        newProperty = ml_model.predict(features)

        # Append this new property vector to the end of df
        new_column = 'Predicted ' + calc_property
        features[new_column] = newProperty

    print('\nfinal performance with multi-feature nested features')
    predict_feature_vector(features, targets)
