import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

print("1. Nạp Hệ thống và Tập Test...")
# Đã xóa chữ 'model/' và 'data/' để khớp với vị trí xuất file của train.py
pipeline_data = joblib.load('model/vietnam_lgbm_pipeline_v2.pkl')
model = pipeline_data['model']

df_test = pd.read_csv('data/Tap_Test_Vietnam_v2.csv')
y_test_real = df_test['price_ThucTe']
X_test = df_test.drop(columns=['price_ThucTe'])

# Khai báo lại kiểu Category cho đúng với bản chất file Test mới
cat_cols = ['province_name', 'district_name', 'ward_name', 'property_type_name', 'house_direction']
for col in cat_cols:
    X_test[col] = X_test[col].astype('category')

print("2. Đang tiến hành dự báo...")
y_pred_log = model.predict(X_test)
y_pred_real = np.expm1(y_pred_log)

print("3. Phân tích Rổ Sai Số...")
sai_so_phan_tram = np.abs((y_test_real - y_pred_real) / y_test_real) * 100

error_df = pd.DataFrame({
    'Giá Thực Tế': y_test_real,
    'Giá Dự Báo': y_pred_real,
    'Sai Số (%)': sai_so_phan_tram
})

# Phân loại rổ sai số
bins = [0, 10, 20, 40, np.inf]
labels = ['Xuất sắc (<10%)', 'Chấp nhận được (10-20%)', 'Kém (20-40%)', 'Sai hoàn toàn (>40%)']
error_df['Phân loại'] = pd.cut(error_df['Sai Số (%)'], bins=bins, labels=labels)

thong_ke = error_df['Phân loại'].value_counts().reset_index()
thong_ke.columns = ['Nhóm Sai Số', 'Số Lượng Căn']
thong_ke['Tỷ Lệ (%)'] = (thong_ke['Số Lượng Căn'] / len(error_df)) * 100

print("\n--- 🏆 KẾT QUẢ ĐÁNH GIÁ (TẬP TEST) ---")
print(thong_ke.to_string(index=False))

# --- VẼ BIỂU ĐỒ ---
plt.figure(figsize=(9, 6))
colors = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c']

# Thêm explode để tách mảng "Xuất sắc" ra một chút cho đẹp mắt
explode = (0.05, 0, 0, 0)

plt.pie(thong_ke['Số Lượng Căn'], labels=thong_ke['Nhóm Sai Số'], autopct='%1.1f%%',
        startangle=140, colors=colors, explode=explode, wedgeprops={'edgecolor': 'white'})

plt.title('Phân phối Độ chính xác - AI Bất Động Sản (LightGBM)', fontsize=14, fontweight='bold', pad=20)

# Căn chỉnh để biểu đồ tròn trịa (không bị méo)
plt.axis('equal')

# Xóa show() nếu chạy file script tự động, thay bằng savefig để lưu ảnh chèn vào báo cáo
plt.savefig('BieuDo_SaiSo_AI.png', dpi=300, bbox_inches='tight')
print("✅ Đã lưu biểu đồ thành file 'BieuDo_SaiSo_AI.png'")