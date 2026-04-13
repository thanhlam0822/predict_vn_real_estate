# 🏡 Hệ Thống Thẩm Định Bất Động Sản Bằng AI (Vietnam Real Estate Valuation)

Dự án này sử dụng Học máy (Machine Learning) với thuật toán **LightGBM** để dự báo và định giá bất động sản tại Việt Nam dựa trên dữ liệu lớn (Big Data). Hệ thống được đóng gói thành một ứng dụng Web trực quan bằng **Streamlit**, cung cấp chiến lược đầu tư thông minh cho người dùng.

---

## ✨ Tính Năng Chính
- **AI Định Giá Cốt Lõi:** Huấn luyện trên gần 1 triệu giao dịch BĐS thực tế, áp dụng các kỹ thuật cao cấp như Log-Transform và Target Encoding để chống nhiễu và xử lý phân phối giá đuôi dài (Long-tail).
- **Phân Tích Đa Chiều:** Đánh giá độ chính xác thông qua rổ sai số (Error Distribution) bằng biểu đồ tròn trực quan.
- **Dynamic Form (Giao diện Thông minh):** Tự động giấu các trường không liên quan (như Mặt tiền, Đường vào) và cập nhật tên nhãn (Label) khi người dùng đổi loại hình BĐS (vd: Chung cư vs. Nhà ngõ hẻm).
- **Tín Hiệu Đầu Tư (Investment Signal):** So sánh giá AI dự báo với giá chủ nhà rao bán để đưa ra lời khuyên: MUA, BỎ QUA, hoặc ĐÀM PHÁN.

---

## 🚀 Kỹ Thuật AI Được Sử Dụng (AI Methodology)
1. **Lọc Dữ Liệu Bằng Đơn Giá:** Sử dụng `price_per_m2` để loại bỏ các tin rác, tin ảo. Xóa cột này trước khi train để tránh Data Leakage.
2. **Log Transformation (`np.log1p`):** Cân bằng lại mô hình trước sự chênh lệch khổng lồ giữa nhà cấp 4 (vài trăm triệu) và siêu biệt thự (hàng trăm tỷ).
3. **Advanced Target Encoding:** Map giá trị trung bình khu vực (Tỉnh/Quận) chỉ trên tập Train để cung cấp cho mô hình một "mỏ neo" giá mà không làm rò rỉ dữ liệu từ tập Test.
4. **Metric Tiêu Chuẩn (MAPE):** Sử dụng sai số phần trăm (MAPE) thay vì RMSE/MAE để phản ánh sát nhất với góc nhìn đầu tư BĐS thực tế.

---

## 📦 Cài Đặt (Installation)

1. Clone kho lưu trữ này về máy:
   ```bash
   https://github.com/thanhlam0822/predict_vn_real_estate.git
   cd predict_vn_real_estate
   
2. Cài đặt thư viện::
   ```bash
   pip install pandas numpy scikit-learn lightgbm joblib matplotlib streamlit fastparquet

## 🛠️ Hướng Dẫn Sử Dụng (Usage)
- **Huấn Luyện:** python train.py (Sinh ra file vietnam_lgbm_pipeline.pkl)
- **Kiểm Thử:** python test_model.py (Vẽ biểu đồ đánh giá sai số)
- **Chạy Web:** streamlit run app.py







