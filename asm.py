import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- TỰ ĐỘNG TẠO FILE MẪU (Để bạn không bị lỗi FileNotFoundError) ---
filename = 'bat_dong_san.csv'
if not os.path.exists(filename):
    data = {
        'gia_nha': [2500, 3000, -500, 4500, 3000, 7000, 15000, None],
        'so_phong': [2, 3, 2, 0, 3, 4, 5, 2],
        'vi_tri': [' Ha Noi', 'hanoi', 'TP HCM', 'Hanoi ', 'Saigon', 'Da Nang', 'HANOI', 'Hue'],
        'tinh_trang': ['Moi', 'Cu', 'Moi', None, 'Moi', 'Cu', 'Moi', 'Moi']
    }
    pd.DataFrame(data).to_csv(filename, index=False)
    print(f"-> Đã tạo file tạm thời: {filename}")

# =====================================================
# 1. KHÁM PHÁ DỮ LIỆU ĐA DẠNG
# =====================================================
df = pd.read_csv(filename)

print("--- Thống kê mô tả ---")
# mean, median, std, min, max
summary = df.describe().T
summary['median'] = df.median(numeric_only=True)
print(summary[['mean', 'median', 'std', 'min', 'max']])

print(f"\nSố lượng giá trị trống:\n{df.isnull().sum()}")

# Vẽ biểu đồ Histogram và Boxplot cho Giá nhà
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['gia_nha'].dropna(), kde=True)
plt.title('Phân phối Giá')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['gia_nha'])
plt.title('Kiểm tra Outliers')
plt.show()

# =====================================================
# 2. XỬ LÝ DỮ LIỆU BẨN
# =====================================================

# 2.1 Điền missing values (Dùng Median cho số, Mode cho chữ)
df['gia_nha'] = df['gia_nha'].fillna(df['gia_nha'].median())
df['tinh_trang'] = df['tinh_trang'].fillna(df['tinh_trang'].mode()[0])

# 2.2 Xử lý dữ liệu không hợp lệ (SỬA LỖI CÚ PHÁP DÒNG 48)
# Loại bỏ giá âm và số phòng phải lớn hơn 0
df = df[(df['gia_nha'] > 0) & (df['so_phong'] > 0)]

# Xử lý Typo (Viết thường và xóa dấu cách thừa)
df['vi_tri'] = df['vi_tri'].str.lower().str.strip()

# 2.3 Loại bỏ dữ liệu trùng lặp
df = df.drop_duplicates()

print("\n--- KẾT QUẢ SAU KHI LÀM SẠCH ---")
print(df)
print("\nThông tin bộ dữ liệu:")
