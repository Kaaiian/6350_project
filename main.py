from Formula import *
from FeatureMatrix import *

formula_list = [['NaCl'], ['Ag'], ['LiAsI']]
formula_column = pd.DataFrame(formula_list)

if __name__ == "__main__":

    element_data = pd.read_csv(r'simple_element_properties.csv', index_col=0)

    feature_matrix = FeatureMatrix()
    column_name = 0
    formula_column = formula_column[column_name]
    for formula in formula_column:
        new_formula = Formula(formula, element_data)
        feature_matrix.addFormula(new_formula.get_feature_vector())
    feature_matrix.createDataFrame()
    print(feature_matrix.get_df_features())
    
    formula_column[0]
