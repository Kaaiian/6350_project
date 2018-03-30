from Formula import *

if __name__ == "__main__":

    element_data = pd.read_csv(r'simple_element_properties.csv', index_col=0)

    formula = "NaCl"
    salt = Formula(formula, element_data)

    print(salt.get_feature_vector())