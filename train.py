import pandas as pd
import numpy as np
import glob
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("1. Đang gộp Data...")
files = glob.glob('shard_*.parquet')
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

print("2. Đang làm sạch dữ liệu (Data Cleaning)...")
df = df[(df['price'] > 300_000_000) & (df['price'] < 200_000_000_000)]
df = df[(df['area'] > 15) & (df['area'] < 1000)]


df['price_per_m2'] = df['price'] / df['area']
df = df[(df['price_per_m2'] >= 15_000_000) & (df['price_per_m2'] <= 800_000_000)]
df = df.drop(columns=['price_per_m2'])

print("3. Kỹ thuật tạo biến (Feature Engineering)...")
df['room_density'] = (df['bedroom_count'] + df['bathroom_count']) / df['area']
df['frontage_ratio'] = df['frontage_width'] / (df['road_width'] + 1)

features = ['area', 'floor_count', 'bedroom_count', 'bathroom_count',
            'frontage_width', 'road_width', 'province_name',
            'district_name', 'ward_name', 'property_type_name', 'house_direction',
            'room_density', 'frontage_ratio']

X = df[features]
y = np.log1p(df['price'])

print("4. Chia tập Train/Test & Tính giá khu vực an toàn...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


y_train_real = np.expm1(y_train)
X_train_temp = X_train.copy()
X_train_temp['price_real'] = y_train_real

dict_province = X_train_temp.groupby('province_name')['price_real'].mean().to_dict()
dict_district = X_train_temp.groupby('district_name')['price_real'].mean().to_dict()
mean_all_train = y_train_real.mean()

X_train['avg_price_province'] = X_train['province_name'].map(dict_province).fillna(mean_all_train)
X_train['avg_price_district'] = X_train['district_name'].map(dict_district).fillna(mean_all_train)

X_test['avg_price_province'] = X_test['province_name'].map(dict_province).fillna(mean_all_train)
X_test['avg_price_district'] = X_test['district_name'].map(dict_district).fillna(mean_all_train)

cat_cols = ['province_name', 'district_name', 'ward_name', 'property_type_name', 'house_direction']
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

print("5. Huấn luyện LightGBM...")
model = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.03, num_leaves=128,
    max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train, categorical_feature=cat_cols)

print("6. Lưu Hệ thống & Tập Test...")

pipeline_data = {
    'model': model,
    'dict_province': dict_province,
    'dict_district': dict_district,
    'mean_price': mean_all_train
}
joblib.dump(pipeline_data, 'vietnam_lgbm_pipeline.pkl')


test_export = X_test.copy()
test_export['price_ThucTe'] = np.expm1(y_test)
test_export.to_csv('Tap_Test_Vietnam.csv', index=False, encoding='utf-8-sig')

print("🎉 Hoàn tất! Đã lưu 'vietnam_lgbm_pipeline.pkl' và 'Tap_Test_Vietnam.csv'")