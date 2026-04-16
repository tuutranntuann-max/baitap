# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("housing.csv")  # đổi tên file của bạn

# =========================
# BÀI 1: SKEWNESS
# =========================
skew = df.select_dtypes(include=np.number).skew().sort_values(ascending=False)

print("=== TOP 10 SKEW ===")
print(skew.head(10))

top3 = skew.head(3).index

for col in top3:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"{col} Distribution")
    plt.show()

# =========================
# BÀI 2: TRANSFORM
# =========================
col1, col2, col3 = top3[0], top3[1], top3[2]

# Log
df[f"{col1}_log"] = np.log1p(df[col1])
df[f"{col2}_log"] = np.log1p(df[col2])

# Box-Cox
df[col1 + "_boxcox"], lam1 = boxcox(df[col1] + 1)
df[col2 + "_boxcox"], lam2 = boxcox(df[col2] + 1)

# Power
pt = PowerTransformer(method='yeo-johnson')
df[col3 + "_power"] = pt.fit_transform(df[[col3]])

# So sánh skew
result = []

for col in [col1, col2, col3]:
    result.append({
        "Column": col,
        "Before": df[col].skew(),
        "Log": df[f"{col}_log"].skew() if f"{col}_log" in df else None,
        "BoxCox": df[f"{col}_boxcox"].skew() if f"{col}_boxcox" in df else None,
        "Power": df[f"{col}_power"].skew() if f"{col}_power" in df else None,
    })

print("\n=== SO SÁNH SKEW ===")
print(pd.DataFrame(result))

# =========================
# BÀI 3: MODEL
# =========================
target = "price"

X = df.select_dtypes(include=np.number).drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model A (raw)
model_A = LinearRegression()
model_A.fit(X_train, y_train)
pred_A = model_A.predict(X_test)

# Model B (log target)
y_train_log = np.log1p(y_train)
model_B = LinearRegression()
model_B.fit(X_train, y_train_log)

pred_B_log = model_B.predict(X_test)
pred_B = np.expm1(pred_B_log)

# Model C (power transform)
pt_model = PowerTransformer()
X_train_pt = pt_model.fit_transform(X_train)
X_test_pt = pt_model.transform(X_test)

model_C = LinearRegression()
model_C.fit(X_train_pt, y_train)
pred_C = model_C.predict(X_test_pt)

# Evaluate
def eval_model(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred)
    )

print("\n=== MODEL RESULT ===")
print("Model A:", eval_model(y_test, pred_A))
print("Model B:", eval_model(y_test, pred_B))
print("Model C:", eval_model(y_test, pred_C))

# =========================
# BÀI 4: VISUAL + INSIGHT
# =========================
colA, colB = col1, col2

# Raw
plt.figure()
sns.histplot(df[colA], kde=True)
plt.title("Raw Data")
plt.show()

# Log
plt.figure()
sns.histplot(np.log1p(df[colA]), kde=True)
plt.title("Log Transform")
plt.show()

# Metric
df["log_price_index"] = np.log1p(df["price"])

print("""
=== INSIGHT ===

- Dữ liệu bị skew làm model kém chính xác
- Log transform giúp dữ liệu cân bằng hơn
- Power transform giảm ảnh hưởng outlier
- Model sau transform thường có RMSE thấp hơn

Ứng dụng:
- Phân nhóm khách hàng theo log-price
- Phát hiện khu vực giá bất thường
- Cải thiện dự đoán giá nhà
""")
