import pandas as pd
import numpy as np

df = pd.read_csv('C:/financeai/data/transactions_data.csv',
                 usecols=['client_id','date'], parse_dates=['date'])

son = df.groupby('client_id')['date'].max()
gun = (pd.Timestamp.now() - son).dt.days

print("Min gün :", gun.min())
print("Max gün :", gun.max())
print("Median  :", gun.median())
print("Unique  :", gun.nunique())
print("\nİlk 10 değer:")
print(gun.value_counts().head(10))