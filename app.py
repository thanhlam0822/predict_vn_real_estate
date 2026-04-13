import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AI Định Giá BĐS Việt Nam", layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load('vietnam_lgbm_pipeline.pkl') # Nhớ sửa lại đường dẫn

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
    if val_str == "" or val_str.isnumeric():
        return False
    return True

st.title("🏡 Hệ Thống Thẩm Định Bất Động Sản Bằng AI")
st.markdown("---")

st.subheader("📝 Thông tin chi tiết bất động sản")

col1, col2 = st.columns(2)

with col1:
    # 1. TỈNH THÀNH
    raw_prov_list = list(hierarchy.keys()) if hierarchy else []
    prov_list = [p for p in raw_prov_list if is_valid_name(p)]
    if not prov_list: prov_list = ["Hồ Chí Minh"]

    # Dùng key để Streamlit tự động quản lý state
    province = st.selectbox("Tỉnh / Thành phố", prov_list, key="sel_prov")

    # 2. QUẬN HUYỆN
    raw_dist_list = list(hierarchy[province].keys()) if (hierarchy and province in hierarchy) else []
    dist_list = [d for d in raw_dist_list if is_valid_name(d)]
    if not dist_list: dist_list = ["Quận 1"]
    district = st.selectbox("Quận / Huyện", dist_list, key="sel_dist")

    # 3. PHƯỜNG XÃ
    raw_ward_list = hierarchy[province][district] if (hierarchy and province in hierarchy and district in hierarchy[province]) else []
    ward_list = [w for w in raw_ward_list if is_valid_name(w)]
    if not ward_list: ward_list = ["Phường Bến Nghé"]
    ward = st.selectbox("Phường / Xã", ward_list, key="sel_ward")

    # 4. LOẠI HÌNH
    prop_type = st.selectbox("Loại hình", ["Căn hộ chung cư", "Nhà ngõ, hẻm", "Nhà mặt phố", "Biệt thự, nhà liền kề"], key="sel_prop")

    # 5. HƯỚNG NHÀ
    house_dir = st.selectbox("Hướng nhà (Ban công/Cửa chính)", ["Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Tây Nam", "Đông Bắc", "Tây Bắc", "KXĐ"], key="sel_dir")

with col2:
    area = st.number_input("Diện tích (m²)", 20.0, 1000.0, 65.0, key="num_area")

    # LOGIC ĐỔI NHÃN CHO TẦNG
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

    # LOGIC ẨN HIỆN MẶT TIỀN (Giữ state an toàn)
    if prop_type == "Căn hộ chung cư":
        st.info("💡 Chung cư không tính Mặt tiền và Đường vào.")
        frontage = 0.0
        road = 0.0
    else:
        frontage = st.number_input("Mặt tiền (m)", 1.0, 50.0, 4.0, key="num_front")
        road = st.number_input("Đường trước nhà (m) - Hẻm/Phố", 1.0, 50.0, 3.0, key="num_road")

st.markdown("---")
gia_rao_ban = st.number_input("💰 Giá chủ nhà rao bán (Tỷ VNĐ) - Để tính chiến lược", 0.0, 500.0, 0.0, key="num_price")

# XỬ LÝ NÚT BẤM
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

    with st.spinner("AI đang phân tích dữ liệu..."):
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

        except Exception as e:
            st.error(f"Có lỗi xảy ra trong quá trình tính toán: {e}")