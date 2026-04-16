# ==============================
# IMPORT THƯ VIỆN
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest


# ==============================
# HÀM HỖ TRỢ
# ==============================
def clean_columns(df):
    df.columns = df.columns.str.lower().str.strip()
    return df


# ==============================
# BÀI 1: HOUSING
# ==============================
print("\n===== BÀI 1: HOUSING =====")

df1 = pd.read_csv("housing.csv")
df1 = clean_columns(df1)

print("Shape:", df1.shape)
print("Missing:\n", df1.isnull().sum())
print(df1.describe())
print("Median:\n", df1.median(numeric_only=True))

# Boxplot
df1.select_dtypes(include=np.number).boxplot(figsize=(10,6))
plt.title("Boxplot Housing")
plt.show()

# Scatter
if 'area' in df1.columns and 'price' in df1.columns:
    plt.scatter(df1['area'], df1['price'])
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Area vs Price")
    plt.show()

# IQR
num_df1 = df1.select_dtypes(include=np.number)
Q1 = num_df1.quantile(0.25)
Q3 = num_df1.quantile(0.75)
IQR = Q3 - Q1

outlier_iqr = ((num_df1 < (Q1 - 1.5 * IQR)) |
               (num_df1 > (Q3 + 1.5 * IQR)))

print("Outlier IQR:\n", outlier_iqr.sum())

# Z-score
z = np.abs(stats.zscore(num_df1.dropna()))
outlier_z = (z > 3)

print("Outlier Z-score:\n", outlier_z.sum())

print("Tổng IQR:", outlier_iqr.sum().sum())
print("Tổng Z-score:", outlier_z.sum().sum())

# Xử lý (clip)
df1_clean = df1.copy()
for col in num_df1.columns:
    df1_clean[col] = np.clip(df1[col],
                            df1[col].quantile(0.05),
                            df1[col].quantile(0.95))

df1_clean.boxplot(figsize=(10,6))
plt.title("Housing After Cleaning")
plt.show()



# ==============================
# BÀI 2: IOT
# ==============================
print("\n===== BÀI 2: IOT =====")

df2 = pd.read_csv("iot.csv", parse_dates=['timestamp'])
df2 = clean_columns(df2)

df2.set_index('timestamp', inplace=True)

print("Missing:\n", df2.isnull().sum())

df2.plot(figsize=(12,6), title="IoT Data")
plt.show()

# Rolling
rolling_mean = df2.rolling(10).mean()
rolling_std = df2.rolling(10).std()

outlier_roll = ((df2 > rolling_mean + 3*rolling_std) |
                (df2 < rolling_mean - 3*rolling_std))

print("Outlier Rolling:\n", outlier_roll.sum())

# Z-score
z2 = np.abs(stats.zscore(df2.dropna()))
print("Outlier Z-score:\n", (z2 > 3).sum())

# Scatter
if 'temperature' in df2.columns and 'pressure' in df2.columns:
    sns.scatterplot(x=df2['temperature'], y=df2['pressure'])
    plt.title("Temp vs Pressure")
    plt.show()

# Clean
df2_clean = df2.interpolate()

df2_clean.plot(title="IoT After Cleaning")
plt.show()



# ==============================
# BÀI 3: E-COMMERCE
# ==============================
print("\n===== BÀI 3: E-COMMERCE =====")

df3 = pd.read_csv("ecommerce.csv")
df3 = clean_columns(df3)

print(df3.describe())
print("Missing:\n", df3.isnull().sum())

cols = [c for c in ['price','quantity','rating'] if c in df3.columns]

# Boxplot
sns.boxplot(data=df3[cols])
plt.title("E-commerce Boxplot")
plt.show()

# IQR
Q1 = df3[cols].quantile(0.25)
Q3 = df3[cols].quantile(0.75)
IQR = Q3 - Q1

outlier3 = ((df3[cols] < (Q1 - 1.5 * IQR)) |
            (df3[cols] > (Q3 + 1.5 * IQR)))

print("Outlier IQR:\n", outlier3.sum())

# Scatter
if 'price' in df3.columns and 'quantity' in df3.columns:
    plt.scatter(df3['price'], df3['quantity'])
    plt.xlabel("Price")
    plt.ylabel("Quantity")
    plt.title("Price vs Quantity")
    plt.show()

# Xử lý
if 'price' in df3.columns:
    df3 = df3[df3['price'] > 0]
    df3['price'] = np.clip(df3['price'],
                           df3['price'].quantile(0.05),
                           df3['price'].quantile(0.95))

if 'rating' in df3.columns:
    df3 = df3[df3['rating'] <= 5]

sns.boxplot(data=df3[cols])
plt.title("After Cleaning")
plt.show()



# ==============================
# BÀI 4: MULTIVARIATE
# ==============================
print("\n===== BÀI 4: MULTIVARIATE =====")

df4 = pd.read_csv("data.csv")
df4 = clean_columns(df4)

num_df4 = df4.select_dtypes(include=np.number).dropna()

model = IsolationForest(contamination=0.05, random_state=42)
df4['outlier'] = model.fit_predict(num_df4)

print(df4['outlier'].value_counts())

# Scatter
if num_df4.shape[1] >= 2:
    sns.scatterplot(x=num_df4.iloc[:,0],
                    y=num_df4.iloc[:,1],
                    hue=df4.loc[num_df4.index, 'outlier'])
    plt.title("Isolation Forest")
    plt.show()
