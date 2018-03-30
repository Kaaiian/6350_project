from Formula import *
from FeatureTargetMatrix import *
from Utility import *


## "add note about removing this." - taylor 
#formula_target_list = [['NaCl', 'Ag', 'LiAsI'], [12, 5, 40]]
#formula_target_column = pd.DataFrame(formula_target_list).transpose()
#formula_target_column.columns = ['formula', 'target']


df = pd.read_excel(r'band_gap-formula-property.xlsx')
df.columns = ['formula', 'target']

#df = df.iloc[0:500]

if __name__ == "__main__":

    element_data = pd.read_csv(r'simple_element_properties.csv', index_col=0)

    feature_matrix = FeatureTargetMatrix()
    series_formula = df['formula']
    series_target = df['target']
    for formula, target in zip(series_formula, series_target):
        new_formula = Formula(formula, target, element_data)
        feature_matrix.addFormula(new_formula.get_feature_vector(), new_formula.get_target())
    feature_matrix.createDataFrame()
#    print(feature_matrix.get_df_features())
#    print(feature_matrix.get_df_targets())

    y_test_list_nest, predicted_test_list_nest = crossValidate(feature_matrix.get_df_features(), feature_matrix.get_df_targets())
    plot_mlOutput(y_test_list_nest, predicted_test_list_nest)

   utput = get_MP_formula_property('Band Gap')
