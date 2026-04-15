import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Định Giá BĐS Việt Nam", layout="wide")


@st.cache_resource
def load_pipeline():
    return joblib.load('model/vietnam_lgbm_pipeline_v2.pkl')


try:
    pipeline = load_pipeline()
    model = pipeline['model']
    dict_province = pipeline['dict_province']
    dict_district = pipeline['dict_district']
    mean_price = pipeline['mean_price']
    hierarchy = pipeline.get('hierarchy', {})
except Exception as e:
    st.error(f"Lỗi: Không thể tải mô hình. Chi tiết: {e}")
    st.stop()


def is_valid_name(name):
    if pd.isna(name) or name is None:
        return False
    val_str = str(name).strip()
    if val_str == "" or val_str.isnumeric() or val_str.lower() == 'nan':
        return False
    return True


st.title("🏡 Hệ Thống Thẩm Định Bất Động Sản Bằng AI")
st.markdown("---")

st.subheader("📝 Thông tin chi tiết bất động sản")

col1, col2 = st.columns(2)

with col1:
    raw_prov_list = list(hierarchy.keys()) if hierarchy else []
    prov_list = [p for p in raw_prov_list if is_valid_name(p)]
    if not prov_list: prov_list = ["Hồ Chí Minh"]
    province = st.selectbox("Tỉnh / Thành phố", prov_list, key="sel_prov")

    raw_dist_list = list(hierarchy.get(province, {}).keys()) if hierarchy else []
    dist_list = [d for d in raw_dist_list if is_valid_name(d)]
    if not dist_list: dist_list = ["Quận 1"]
    district = st.selectbox("Quận / Huyện", dist_list, key="sel_dist")

    raw_ward_list = hierarchy.get(province, {}).get(district, []) if hierarchy else []
    ward_list = [w for w in raw_ward_list if is_valid_name(w)]
    if not ward_list: ward_list = ["Phường Bến Nghé"]
    ward = st.selectbox("Phường / Xã", ward_list, key="sel_ward")

    prop_type = st.selectbox("Loại hình", ["Căn hộ chung cư", "Nhà ngõ, hẻm", "Nhà mặt phố", "Biệt thự, nhà liền kề"],
                             key="sel_prop")
    house_dir = st.selectbox("Hướng nhà (Ban công/Cửa chính)",
                             ["Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Tây Nam", "Đông Bắc", "Tây Bắc", "KXĐ"],
                             key="sel_dir")

with col2:
    area = st.number_input("Diện tích (m²)", 20.0, 1000.0, 65.0, key="num_area")

    if prop_type == "Căn hộ chung cư":
        floor_label = "Vị trí tầng (Để số 8 nếu không rõ)"
        floor_default = 8
    else:
        floor_label = "Tổng số tầng của Căn nhà"
        floor_default = 3
    floor = st.number_input(floor_label, 1, 100, floor_default, key="num_floor")

    c_bed, c_bath = st.columns(2)
    with c_bed:
        bed = st.number_input("Phòng ngủ", 1, 10, 2, key="num_bed")
    with c_bath:
        bath = st.number_input("Phòng vệ sinh", 1, 10, 2, key="num_bath")

    if prop_type == "Căn hộ chung cư":
        st.info("💡 Chung cư không tính Mặt tiền và Đường vào.")
        frontage = 0.0
        road = 0.0
    else:
        frontage = st.number_input("Mặt tiền (m)", 1.0, 50.0, 4.0, key="num_front")
        road = st.number_input("Đường trước nhà (m) - Hẻm/Phố", 1.0, 50.0, 3.0, key="num_road")

st.markdown("---")
gia_rao_ban = st.number_input("💰 Giá chủ nhà rao bán (Tỷ VNĐ) - Nhập để AI tư vấn chiến lược", 0.0, 500.0, 0.0,
                              key="num_price")

if st.button("Thẩm Định Ngay", use_container_width=True):
    room_density = (bed + bath) / area
    frontage_ratio = frontage / (road + 1)

    avg_prov = dict_province.get(province, mean_price)
    avg_dist = dict_district.get(district, mean_price)

    input_df = pd.DataFrame({
        'area': [area], 'floor_count': [floor], 'bedroom_count': [bed], 'bathroom_count': [bath],
        'frontage_width': [frontage], 'road_width': [road], 'province_name': [province],
        'district_name': [district], 'ward_name': [ward], 'property_type_name': [prop_type],
        'house_direction': [house_dir], 'room_density': [room_density], 'frontage_ratio': [frontage_ratio],
        'avg_price_province': [avg_prov], 'avg_price_district': [avg_dist]
    })

    cat_cols = ['province_name', 'district_name', 'ward_name', 'property_type_name', 'house_direction']
    for col in cat_cols:
        input_df[col] = input_df[col].astype('category')

    with st.spinner("AI đang phân tích hàng triệu dữ liệu thị trường..."):
        try:
            pred_log = model.predict(input_df)[0]
            pred_real = np.expm1(pred_log) / 1e9

            mape_margin = 0.22
            bien_duoi = pred_real * (1 - mape_margin)
            bien_tren = pred_real * (1 + mape_margin)

            st.markdown("## 📊 KẾT QUẢ ĐỊNH GIÁ")
            c1, c2, c3 = st.columns(3)
            c1.metric("Biên Dưới (Mua Tốt)", f"{bien_duoi:.2f} Tỷ")
            c2.metric("GIÁ TRỊ THỰC TẾ (AI)", f"{pred_real:.2f} Tỷ")
            c3.metric("Biên Trên (Rủi Ro)", f"{bien_tren:.2f} Tỷ")

            if gia_rao_ban > 0:
                st.markdown("### 💡 Tín Hiệu Đầu Tư")
                do_lech = ((gia_rao_ban - pred_real) / pred_real) * 100

                if gia_rao_ban < bien_duoi:
                    st.success(f"✅ TÍN HIỆU MUA: Đang bán rẻ hơn thị trường {abs(do_lech):.1f}%. Cơ hội chốt lời!")
                elif gia_rao_ban > bien_tren:
                    st.error(f"❌ TÍN HIỆU BỎ QUA: Giá đang cao hơn thị trường {do_lech:.1f}%. Rủi ro đọng vốn.")
                else:
                    st.warning(f"⚠️ TÍN HIỆU ĐÀM PHÁN: Giá khá sát thị trường. Ép giá quanh mốc {bien_duoi:.2f} Tỷ.")

            st.markdown("---")
            st.subheader("🔍 Tại sao AI đưa ra mức giá này?")
            st.info("Biểu đồ thể hiện **Tỷ lệ phần trăm (%) đóng góp** của từng yếu tố vào quyết định định giá.")

            importance_gain = model.booster_.feature_importance(importance_type='gain')
            feature_names = model.booster_.feature_name()
            importance_percent = (importance_gain / importance_gain.sum()) * 100

            df_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_percent
            })

            dict_labels = {
                'avg_price_district': 'Giá trung bình Quận/Huyện',
                'area': 'Diện tích (m²)',
                'avg_price_province': 'Giá trung bình Tỉnh/Thành',
                'ward_name': 'Phường/Xã',
                'district_name': 'Quận/Huyện',
                'floor_count': 'Số tầng / Vị trí tầng',
                'road_width': 'Đường trước nhà',
                'frontage_width': 'Mặt tiền',
                'room_density': 'Mật độ phòng (Nhồi nhét)',
                'frontage_ratio': 'Tỷ lệ Vàng (Mặt tiền/Đường)',
                'house_direction': 'Hướng nhà',
                'property_type_name': 'Loại hình BĐS'
            }
            df_importance['Feature'] = df_importance['Feature'].map(dict_labels).fillna(df_importance['Feature'])
            df_importance = df_importance.sort_values(by='Importance', ascending=True).tail(10)

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(df_importance['Feature'], df_importance['Importance'], color='#3498db', edgecolor='black',
                           alpha=0.8)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{width:.1f}%',
                        va='center', ha='left', fontsize=11, fontweight='bold', color='#2c3e50')

            ax.set_xlabel('Tỷ trọng quyết định (%)', fontsize=12, fontweight='bold')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()

        except Exception as e:
            st.error(f"Có lỗi xảy ra trong quá trình dự báo: {e}")
