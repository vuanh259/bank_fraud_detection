import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. CẤU HÌNH GIAO DIỆN WEB
st.set_page_config(page_title="VietinBank Fraud Detection Sandbox", page_icon="🏦", layout="centered")

st.title("🏦 Hệ thống Thử nghiệm Phát hiện Gian lận Hồ sơ")
st.markdown("---")

# 2. TẢI MÔ HÌNH VÀ CẤU TRÚC ĐẶC TRƯNG ĐÃ LƯU
@st.cache_resource # Giúp lưu bộ nhớ đệm, không phải load lại file mỗi lần bấm nút
def load_assets():
    model = joblib.load("rf_fraud_model.pkl")
    model_features = joblib.load("model_features.pkl")
    return model, model_features

try:
    model, model_features = load_assets()
    st.success("🤖 Hệ thống AI: Mô hình phòng chống rủi ro đã Sẵn sàng vận hành!")
except:
    st.error("❌ Không tìm thấy file mô hình. Vui lòng chạy Cell 12 trong file Notebook trước!")
    st.stop()

st.subheader("📝 Nhập thông tin hồ sơ kiểm tra")

# 3. TẠO FORM NHẬP LIỆU CÁC CHỈ SỐ CỐT LÕI (Dựa trên Top Feature Importance của bạn)
# Chúng ta chọn ra các chỉ số quan trọng nhất để người dùng tự điền trên giao diện
customer_age = st.slider("1. Độ tuổi khách hàng (customer_age):", min_value=18, max_value=100, value=30)
keep_alive_session = st.number_input("2. Thời gian giữ session thao tác (keep_alive_session):", min_value=0.0, value=15.5)
bank_months_count = st.slider("3. Thời gian gắn bó với bank cũ - tháng (bank_months_count):", min_value=-1, max_value=36, value=12)
income = st.slider("4. Chỉ số thu nhập (income):", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
current_address_months_count = st.number_input("5. Thời gian ở địa chỉ hiện tại - tháng:", min_value=-1, value=24)

# Nhập các thông tin dạng phân loại (Categorical)
device_os = st.selectbox("6. Hệ điều hành thiết bị đăng ký:", ["Windows", "iOS", "Android", "Other"])
payment_type = st.selectbox("7. Hình thức thanh toán lựa chọn:", ["AC", "AB", "AD", "Other_Type"])
phone_home_valid = st.selectbox("8. Trạng thái xác thực số điện thoại bàn:", ["Hợp lệ (1)", "Không hợp lệ (0)"])

# Ngưỡng vận hành tối ưu toán học đã tìm được ở Cell 9
OPTIMAL_THRESHOLD = 0.2400 

# 4. XỬ LÝ DỮ LIỆU KHI NGƯỜI DÙNG BẤM NÚT KIỂM TRA
if st.button("🔍 Tiến hành Thẩm định Hồ sơ"):
    
    # Tạo một từ điển dữ liệu thô từ các input của người dùng
    raw_input = {
        'customer_age': customer_age,
        'keep_alive_session': keep_alive_session,
        'bank_months_count': bank_months_count,
        'income': income,
        'current_address_months_count': current_address_months_count,
        'phone_home_valid': 1 if "Hợp lệ" in phone_home_valid else 0
    }
    
    # Tự động map các biến One-Hot Encoding dựa trên selectbox
    raw_input['device_os_windows'] = 1 if device_os == "Windows" else 0
    raw_input['device_os_other'] = 1 if device_os == "Other" else 0
    raw_input['payment_type_AC'] = 1 if payment_type == "AC" else 0
    raw_input['payment_type_AB'] = 1 if payment_type == "AB" else 0

    # Khởi tạo một hàng DataFrame trống với đầy đủ các cột giống hệt tập Train (X)
    input_df = pd.DataFrame(0, index=[0], columns=model_features)
    
    # Điền các giá trị người dùng đã nhập vào đúng tên cột tương ứng
    for col in raw_input:
        if col in input_df.columns:
            input_df[col] = raw_input[col]
            
    # 5. ĐƯA DỮ LIỆU VÀO MÔ HÌNH ĐỂ DỰ ĐOÁN XÁC SUẤT
    fraud_probability = model.predict_proba(input_df)[0, 1]
    
    # 6. HIỂN THỊ KẾT QUẢ DỰA TRÊN NGƯỠNG TỐI ƯU
    st.markdown("---")
    st.subheader("📊 Kết quả phân tích rủi ro")
    st.metric(label="Xác suất nghi ngờ Gian lận từ AI", value=f"{fraud_probability * 100:.2f}%")
    
    if fraud_probability >= OPTIMAL_THRESHOLD:
        st.error(f"🚨 CẢNH BÁO ĐỎ: Hồ sơ này bị hệ thống đánh giá có dấu hiệu GIAN LẬN! (Vượt ngưỡng an toàn {OPTIMAL_THRESHOLD * 100}%)")
        st.markdown("""
        **Khuyến nghị nghiệp vụ cho nhân viên thẩm định:**
        * Tạm dừng phê duyệt tự động luồng eKYC.
        * Yêu cầu khách hàng xác thực trực tiếp hoặc thực hiện cuộc gọi video (Video Call Verify).
        * Kiểm tra kỹ sự trùng khớp về mặt địa lý giữa thiết bị và địa chỉ đăng ký.
        """)
    else:
        st.success(f"✅ HỒ SƠ AN TOÀN: Xác suất rủi ro nằm trong phạm vi cho phép (Dưới ngưỡng quyết định {OPTIMAL_THRESHOLD * 100}%).")
        st.markdown("**Khuyến nghị nghiệp vụ:** Cho phép hồ sơ chuyển tiếp sang bước phê duyệt tiếp theo tự động.")