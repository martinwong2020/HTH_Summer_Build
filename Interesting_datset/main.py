import pandas

pandas.options.display.max_columns = None
pandas.options.display.max_rows = None

csv_data = pandas.read_csv("Interesting_datset/supermarket_sales.csv")
print(csv_data)