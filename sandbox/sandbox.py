import pandas as pd
from tabulate import tabulate
import numpy as np

# Import dataset
df = pd.read_csv("telco_sandbox_data.csv")

# Print out your training data in a table
print('before: \n')
print(df.head)
# print('before\n', tabulate(df, tablefmt="psql"))

# Try out stuff here to adjust df
#
#
#
# #

# Print out your training data in a table
print('after: \n')
print(df.head)
# print('\nafter\n', tabulate(df, tablefmt="psql"))


