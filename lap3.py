import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Cấu hình để vẽ biểu đồ tiếng Việt không bị lỗi font (nếu cần)
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 5)

    # Cấu hình đúng tên file và tên cột lấy từ dữ liệu thực tế của Tuấn
tasks = [
        {
            'file': 'ITA105_Lab_3_Sports.csv', 
            'col': 'toc_do_100m_s', 
            'title': 'Bai 1: Van dong vien'
        },
        {
            'file': 'ITA105_Lab_3_Health.csv', 
            'col': 'huyet_ap_mmHg', 
            'title': 'Bai 2: Chi so benh nhan'
        },
        {
            'file': 'ITA105_Lab_3_Finance.csv', 
            'col': 'doanh_thu_musd', 
            'title': 'Bai 3: Chi so cong ty'
        },
        {
            'file': 'ITA105_Lab_3_Gaming.csv', 
            'col': 'gio_choi', 
            'title': 'Bai 4: Nguoi choi truc tuyen'
        }
    ]

for t in tasks:
        print(f"\n>>> Dang xu ly {t['title']}...")
        try:
            # 1. Doc du lieu
            df = pd.read_csv(t['file'])
            
            # 2. Kiem tra gia tri thieu
            print(f"- Gia tri thieu: {df[t['col']].isnull().sum()}")
            
            # 3. Chuan hoa
            data_col = df[[t['col']]]
            df['MinMax'] = MinMaxScaler().fit_transform(data_col)
            df['ZScore'] = StandardScaler().fit_transform(data_col)

            # 4. Ve bieu do so sanh phan phoi
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            sns.histplot(df[t['col']], kde=True, ax=axes[0], color='blue').set_title('Goc (Original)')
            sns.histplot(df['MinMax'], kde=True, ax=axes[1], color='green').set_title('Min-Max [0,1]')
            sns.histplot(df['ZScore'], kde=True, ax=axes[2], color='red').set_title('Z-Score (mean=0)')
            plt.suptitle(t['title'])
            plt.show()

            # 5. Ve Boxplot de tim ngoai le (Outliers)
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[t['col']], color='yellow')
            plt.title(f"Boxplot phat hien ngoai le - {t['col']}")
            plt.show()

            # Rieng Bai 3: Ve Scatterplot giua Doanh thu va Loi nhuan
            if t['file'] == 'ITA105_Lab_3_Finance.csv':
                print("- Dang ve Scatterplot cho Bai 3...")
                df['Profit_Scaled'] = StandardScaler().fit_transform(df[['loi_nhuan_musd']])
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                sns.scatterplot(data=df, x='doanh_thu_musd', y='loi_nhuan_musd', ax=ax[0]).set_title('Truoc chuan hoa')
                sns.scatterplot(data=df, x='ZScore', y='Profit_Scaled', ax=ax[1], color='orange').set_title('Sau Z-Score')
                plt.show()

        except FileNotFoundError:
            print(f"Loi: Khong tim thay file {t['file']}. Hay kiem tra duong dan!")
        except KeyError:
            print(f"Loi: Khong tim thay cot '{t['col']}' trong file {t['file']}.")
