import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('NZDUSD_H4.csv', nrows=2500)


df = df.drop(['SMA14IND', 'SMA50IND'], axis=1)


missing_values_count = df['Close'].isnull().sum()
print(f"Missing data in 'Close' column: {missing_values_count}")


df['Close'] = df['Close'].interpolate()


df['SMA14'] = df['SMA14'].fillna(df['SMA14'].mean())
df['SMA50'] = df['SMA50'].fillna(df['SMA50'].mean())

# Catch-all: Filling any remaining missing data with zeros



correlation_sma14_sma50 = df['SMA14'].corr(df['SMA50'])
print(f"Correlation between SMA14 and SMA50: {correlation_sma14_sma50:.4f}")

correlation_close_sma14 = df['Close'].corr(df['SMA14'])
print(f"Correlation between Close and SMA14: {correlation_close_sma14:.4f}")

correlation_close_sma50 = df['Close'].corr(df['SMA50'])
print(f"Correlation between Close and SMA50: {correlation_close_sma50:.4f}")

if correlation_close_sma14 > correlation_close_sma50:
    print("Action: Dropping SMA50 (weaker correlation with Close).")
    df = df.drop(['SMA50'], axis=1)
else:
    print("Action: Dropping SMA14 (weaker correlation with Close).")
    df = df.drop(['SMA14'], axis=1)


negative_cci_count = (df['CCI'] < 0).sum()
print(f"Number of negative elements in the CCI attribute: {negative_cci_count}")

attribute_info = df.describe().transpose()[['min', 'max']]
print("\n--- Attribute Min/Max Information ---")
print(attribute_info)


attributes = ['Close', 'CCI']
for attribute in attributes:
    max_val = df[attribute].max()
    min_val = df[attribute].min()
    

    if max_val != min_val:
        df[attribute] = (df[attribute] - min_val) / (max_val - min_val)


labels_2categories = ['Low', 'High']
labels_4categories = ['Very Low', 'Low', 'High', 'Very High']

df['Close_2categories'] = pd.cut(df['Close'], bins=2, labels=labels_2categories)
df['Close_4categories'] = pd.cut(df['Close'], bins=4, labels=labels_4categories)

df['DM_2categories'] = pd.cut(df['DM'], bins=2, labels=labels_2categories)
df['DM_4categories'] = pd.cut(df['DM'], bins=4, labels=labels_4categories)


print("\n--- First rows with categorical features ---")
print(df[['Close', 'Close_2categories', 'Close_4categories']].head())


plt.figure(figsize=(8, 8))
df['Decision'].value_counts().plot.pie(autopct='%1.1f%%', title='Decision Distribution')
plt.ylabel('') 
plt.show()

plt.figure(figsize=(12, 6))
df['Close'].plot(title='Normalized Close Price (NZD/USD)', color='blue')
plt.xlabel('Time (Index)')
plt.ylabel('Normalized Price')
plt.grid(True)
plt.show()

# Final Sanity Check
print("\n--- Final Dataset Info ---")
print(df.info())
