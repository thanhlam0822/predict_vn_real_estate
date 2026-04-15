import pandas as pd
import numpy as np
import glob
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

# ==========================================
# 1. NẠP DỮ LIỆU TỪ PARQUET (Tự động gom file)
# ==========================================
print("1. Đang nạp dữ liệu từ các file Parquet...")
files = glob.glob('/kaggle/input/datasets/nguyenphatq/vietname-real-estate-v2/shard_*.parquet')
if not files:
    print("Lỗi: Không tìm thấy file shard_*.parquet nào!")
    exit()

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# ==========================================
# 2. LÀM SẠCH VÀ CHUẨN HÓA KIỂU DỮ LIỆU
# ==========================================
print("2. Đang chuẩn hóa kiểu dữ liệu và làm sạch...")

# Ép kiểu toàn bộ các cột định lượng từ String về dạng Số (Float/Int)
numeric_columns = ['price', 'area', 'floor_count', 'bedroom_count', 'bathroom_count', 'frontage_width', 'road_width']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Bắt đầu lọc dữ liệu (Bây giờ máy tính đã hiểu price và area là các con số)
df = df[(df['price'] >= 300_000_000) & (df['price'] <= 200_000_000_000)]
df = df[(df['area'] >= 15) & (df['area'] <= 1000)]

# Lọc bằng đơn giá (chống rò rỉ dữ liệu)
df['price_per_m2'] = df['price'] / df['area']
df = df[(df['price_per_m2'] >= 15_000_000) & (df['price_per_m2'] <= 800_000_000)]
df = df.drop(columns=['price_per_m2'])

# Lọc bỏ các dòng bị khuyết thông tin vị trí trọng yếu
df = df.dropna(subset=['province_name', 'district_name', 'property_type_name'])

# ==========================================
# 3. KỸ THUẬT TẠO BIẾN (FEATURE ENGINEERING)
# ==========================================
print("3. Kỹ thuật tạo biến (Feature Engineering)...")
# Điền tạm số 1 hoặc 0 cho các phòng/đường bị trống để tránh lỗi chia cho 0
df['room_density'] = (df['bedroom_count'].fillna(1) + df['bathroom_count'].fillna(1)) / df['area']
df['frontage_ratio'] = df['frontage_width'].fillna(0) / (df['road_width'].fillna(0) + 1)

features_cols = ['area', 'floor_count', 'bedroom_count', 'bathroom_count',
                 'frontage_width', 'road_width', 'province_name',
                 'district_name', 'ward_name', 'property_type_name', 'house_direction',
                 'room_density', 'frontage_ratio']

X = df[features_cols]
y = np.log1p(df['price']) # Trị phân phối đuôi dài

# ==========================================
# 4. CHIA TẬP TRAIN/TEST & TARGET ENCODING
# ==========================================
print("4. Chia tập Train/Test & Xử lý Target Encoding...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_real = np.expm1(y_train)
X_train_temp = X_train.copy()
X_train_temp['price_real'] = y_train_real

# Tạo mỏ neo giá khu vực
dict_province = X_train_temp.groupby('province_name')['price_real'].mean().to_dict()
dict_district = X_train_temp.groupby('district_name')['price_real'].mean().to_dict()
mean_all_train = y_train_real.mean()

X_train['avg_price_province'] = X_train['province_name'].map(dict_province).fillna(mean_all_train)
X_train['avg_price_district'] = X_train['district_name'].map(dict_district).fillna(mean_all_train)
X_test['avg_price_province'] = X_test['province_name'].map(dict_province).fillna(mean_all_train)
X_test['avg_price_district'] = X_test['district_name'].map(dict_district).fillna(mean_all_train)

# ==========================================
# 5. XỬ LÝ KIỂU DỮ LIỆU BẢN ĐỊA CHO LIGHTGBM
# ==========================================
cat_cols = ['province_name', 'district_name', 'ward_name', 'property_type_name', 'house_direction']
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# ==========================================
# 6. TẠO CÂY ĐỊA GIỚI (HIERARCHY CHO WEB)
# ==========================================
print("5. Tạo Cây địa giới cho Giao diện Web...")
hierarchy = {}
for prov in df['province_name'].unique():
    districts = df[df['province_name'] == prov]['district_name'].unique()
    hierarchy[prov] = {}
    for dist in districts:
        wards = df[(df['province_name'] == prov) & (df['district_name'] == dist)]['ward_name'].dropna().unique().tolist()
        hierarchy[prov][dist] = wards

# ==========================================
# 7. HUẤN LUYỆN LIGHTGBM
# ==========================================
print("6. Đang huấn luyện AI (LightGBM)...")
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=128,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, categorical_feature=cat_cols)

y_pred_log = model.predict(X_test)
mape = np.mean(np.abs((np.expm1(y_test) - np.expm1(y_pred_log)) / np.expm1(y_test))) * 100
print(f"🔥 MAPE trên tập Test: {mape:.2f}%")

# ==========================================
# 8. ĐÓNG GÓI VÀ XUẤT FILE
# ==========================================
print("7. Đóng gói mô hình...")
pipeline_data = {
    'model': model,
    'dict_province': dict_province,
    'dict_district': dict_district,
    'mean_price': mean_all_train,
    'hierarchy': hierarchy
}
joblib.dump(pipeline_data, 'vietnam_lgbm_pipeline_v2.pkl')

test_export = X_test.copy()
test_export['price_ThucTe'] = np.expm1(y_test)
test_export.to_csv('Tap_Test_Vietnam_v2.csv', index=False, encoding='utf-8-sig')
print("✅ Xong! Đã xuất vietnam_lgbm_pipeline.pkl và Tap_Test_Vietnam.csv")