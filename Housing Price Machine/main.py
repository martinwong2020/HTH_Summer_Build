import pandas as pd
pd.options.display.max_columns=None
pd.options.display.max_rows=None

housing_data=pd.read_csv("Housing Price Machine/housing.csv")
print(housing_data)