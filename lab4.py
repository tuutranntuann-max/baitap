#PHẦN 1 — Import thư viện

# Import pandas để xử lý dữ liệu bảng
import pandas as pd

# Import numpy để xử lý số
import numpy as np

# Import LinearRegression từ thư viện sklearn
from sklearn.linear_model import LinearRegression

# Import thư viện vẽ biểu đồ
import matplotlib.pyplot as plt

#PHẦN 2 — Tạo dataset

# Tạo dataset bằng dictionary
data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Score": [2,4,5,6,7,8,8.5,9]
}

# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Hiển thị dataset
df

# PHẦN 3 — Tách Input và Output

#Trong Machine Learning:
# Input = Feature
# Output = Label

# X là input (feature)
X = df[["Hours"]]

# y là output (label)
y = df["Score"]

# Giải thích:
# Hours -> input
# Score -> output

#PHẦN 4 — Tạo Machine Learning Model

# Tạo model Linear Regression
model = LinearRegression()  #model sẽ học mối quan hệ giữa Hours và Score

# PHẦN 5 — Train Model

# fit() dùng để train model
# model sẽ học từ dataset
model.fit(X, y)
# model đã học được quy luật

#PHẦN 6 — Dự đoán dữ liệu mới
# Tạo dữ liệu mới
new_hours = [[6]]

# Predict dùng để dự đoán
predicted_score = model.predict(new_hours)

# In kết quả
print("Predicted score:", predicted_score)


# PHẦN 7 — Dự đoán nhiều giá trị

# Tạo nhiều giá trị hours
new_data = [[4],[6],[9]]

# Dự đoán điểm
predictions = model.predict(new_data)

# Hiển thị kết quả
print(predictions)

#PHẦN 8 — Vẽ biểu đồ dữ liệu
# Vẽ các điểm dữ liệu
plt.scatter(X, y)

# Vẽ đường dự đoán
plt.plot(X, model.predict(X))

# Nhãn trục
plt.xlabel("Hours studied")
plt.ylabel("Score")

# Tiêu đề
plt.title("Hours vs Score")

# Hiển thị biểu đồ
plt.show()

# PHẦN 9 — Đánh giá Model
# Import hàm đánh giá
from sklearn.metrics import r2_score

# Dự đoán trên dataset
y_pred = model.predict(X)

# Tính R2 score
score = r2_score(y, y_pred)

print("R2 Score:", score)
