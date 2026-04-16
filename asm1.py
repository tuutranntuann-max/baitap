# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("housing.csv")   # đổi tên file của bạn

print("Shape:", df.shape)
print(df.head())


# =====================================================
# 1. KHÁM PHÁ DỮ LIỆU (EDA)
# =====================================================

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLICATES ===")
print(df.duplicated().sum())

print("\n=== THỐNG KÊ ===")
print(df.describe())

# Histogram
df.hist(figsize=(12,8))
plt.show()

# Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(data=df.select_dtypes(include=np.number))
plt.xticks(rotation=90)
plt.show()

# Violin plot
plt.figure(figsize=(12,6))
sns.violinplot(data=df.select_dtypes(include=np.number))
plt.xticks(rotation=90)
plt.show()

# Categorical distribution
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    print(f"\nValue counts - {col}")
    print(df[col].value_counts().head())


# =====================================================
# 2. XỬ LÝ DỮ LIỆU BẨN
# =====================================================

# Fill missing
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove invalid
if "price" in df.columns:
    df = df[df["price"] > 0]

if "rooms" in df.columns:
    df = df[df["rooms"] > 0]

# Remove duplicates
df = df.drop_duplicates()

print("\nSau khi clean:", df.shape)


# =====================================================
# 3. OUTLIER & SKEW
# =====================================================

if "price" in df.columns:
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1

    # Remove outlier
    df = df[(df["price"] > Q1 - 1.5*IQR) & (df["price"] < Q3 + 1.5*IQR)]

# Skew
print("\nSkewness:")
print(df.select_dtypes(include=np.number).skew())


# =====================================================
# 4. CHUẨN HÓA & ENCODING
# =====================================================

# Scaling
scaler = StandardScaler()
num_cols = df.select_dtypes(include=np.number).columns

df[num_cols] = scaler.fit_transform(df[num_cols])

# Encoding categorical
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("\nĐã encode categorical")


# =====================================================
# 5. TEXT PROCESSING (TF-IDF)
# =====================================================

if "description" in df.columns:
    df["description"] = df["description"].astype(str)

    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(df["description"])

    print("\nTF-IDF shape:", tfidf_matrix.shape)


# =====================================================
# 6. PHÁT HIỆN DUPLICATE BẰNG TEXT SIMILARITY
# =====================================================

if "description" in df.columns:
    similarity = cosine_similarity(tfidf_matrix)

    duplicates = []

    for i in range(len(similarity)):
        for j in range(i+1, len(similarity)):
            if similarity[i][j] > 0.8:
                duplicates.append((i, j))

    print("\nCác bản ghi giống nhau (text similarity > 0.8):")
    print(duplicates[:10])


# =====================================================
# DONE
# =====================================================

print("\nHOÀN THÀNH GIAI ĐOẠN 1")
