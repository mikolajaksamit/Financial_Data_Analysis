import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Reading data from a CSV file containing financial data (NZD/USD).

# Limiting the number of rows to 2500 for efficiency.

df= pd.read_csv('NZDUSD_H4.csv')

df = df.head(2500)

# Dropping two columns: 'SMA14IND' and 'SMA50IND'.
df = df.drop(['SMA14IND', 'SMA50IND'], axis=1)

# Checking the number of missing values in the 'Close' column.
missing_values_count = df['Close'].isnull().sum()
print(f'Suma pustych danych w kolumnie close: {missing_values_count}')

# Interpolating missing data in the 'Close' column.
df['Close'] = df['Close'].interpolate()

# Filling missing data in the 'SMA14' and 'SMA50' columns with the mean value.

df['SMA14'] = df['SMA14'].fillna(df['SMA14'].mean())
df['SMA50'] = df['SMA50'].fillna(df['SMA50'].mean())

# Filling remaining missing data with zeros.
df= df.fillna(0)

# Calculating the correlation between the 'SMA14' and 'SMA50' columns.
correlation_sma14_sma50 = df['SMA14'].corr(df['SMA50'])

print(f'Correlation between SMA14 and SMA50: {correlation_sma14_sma50}')

# Calculating the correlation between the 'Close' column and 'SMA14' as well as 'SMA50'.

correlation_close_sma14 = df['Close'].corr(df['SMA14'])
print(f'Correlation between Close and SMA14: {correlation_close_sma14}')

correlation_close_sma50 = df['Close'].corr(df['SMA50'])
print(f'Correlation between Close and SMA50: {correlation_close_sma50}')

# Dropping one of the columns ('SMA14' or 'SMA50') based on higher correlation with 'Close'.

if (correlation_close_sma14) > (correlation_close_sma50):
  df = df.drop(['SMA14'], axis=1)
else:
  df = df.drop(['SMA50'], axis=1)

# Counting the number of negative elements in the 'CCI' attribute.
negative_cci_count = (df['CCI'] < 0).sum()
print(f'Number of negative elements in the CCI attribute: {negative_cci_count}')

# Displaying information about minimum and maximum values for each attribute.
attribute_info = df.describe().transpose()[['min', 'max']]
print(attribute_info)

# Normalizing two selected attributes: 'Close' and 'CCI'.
atrybuty=['Close', 'CCI']
for atrybut in atrybuty:
  max = df[atrybut].max()
  min = df[atrybut].min()
  df[atrybut] = (df[atrybut] - min) / (max - min)

# Discretizing the 'Close' and 'DM' attributes into two and four categories.
# Results are saved in new columns 'Close_2categories', 'Close_4categories', 'DM_2categories', and 'DM_4categories'.

etykiety_2kategorie = ['Low', 'High']
etykiety_4kategorie = ['Very Low', 'Low', 'High', 'Very High']

seria_2kategorie = pd.cut(df['Close'], 2, labels=etykiety_2kategorie)
seria_4kategorie = pd.cut(df['Close'], 4, labels=etykiety_4kategorie)
df['Close_2categories']=seria_2kategorie
df['Close_4categories']=seria_4kategorie
seria_2kategorie1 = pd.cut(df['DM'], 2, labels=etykiety_2kategorie)
seria_4kategorie1= pd.cut(df['DM'], 4, labels=etykiety_4kategorie)
df['DM_2categories']=seria_2kategorie1
df['DM_4categories']=seria_4kategorie1

# Displaying the first rows with the new discretization columns.

print(df[['Close', 'Close_2categories', 'Close_4categories']].head())
print(df[['DM', 'DM_2categories', 'DM_4categories']].head())
# Pie chart for visualizing the distribution of values in the 'Decision' column.

df['Decision'].value_counts().plot.pie()

# Line plot for the 'Close' column.

df['Close'].plot()
#plt.show

# Displaying information about the data.

print(df.info())
print(df.head())


