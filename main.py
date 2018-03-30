from Formula import *
from FeatureTargetMatrix import *


# "add note about removing this." - taylor 
formula_target_list = [['NaCl', 'Ag', 'LiAsI'], [12, 5, 40]]
formula_target_column = pd.DataFrame(formula_target_list).transpose()
formula_target_column.columns = ['formula', 'target']






if __name__ == "__main__":

    element_data = pd.read_csv(r'simple_element_properties.csv', index_col=0)

    feature_matrix = FeatureTargetMatrix()
    series_formula = formula_target_column['formula']
    series_target = formula_target_column['target']
    for formula, target in zip(series_formula, series_target):
        new_formula = Formula(formula, target, element_data)
        feature_matrix.addFormula(new_formula.get_feature_vector(), new_formula.get_target())
    feature_matrix.createDataFrame()
    print(feature_matrix.get_df_features())
    print(feature_matrix.get_df_targets())
