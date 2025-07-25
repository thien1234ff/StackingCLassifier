import streamlit as st
import numpy as np
import joblib

# Load mô hình stacking
model = joblib.load("stacking_model.pkl")

# Giao diện chính
st.title("🌱 Hệ thống đề xuất cây trồng bằng Stacking Classifier")

st.markdown("### Nhập các thông số đầu vào:")

# Các biến đầu vào
n = st.slider("N - Nitơ", 0, 140, 90)
p = st.slider("P - Phốt pho", 0, 140, 42)
k = st.slider("K - Kali", 0, 200, 43)
ph = st.number_input("pH đất", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
temperature = st.number_input("Nhiệt độ (°C)", value=25.0)
humidity = st.number_input("Độ ẩm (%)", value=80.0)
rainfall = st.number_input("Lượng mưa (mm)", value=100.0)

# Dự đoán
if st.button("🌾 Dự đoán cây trồng phù hợp"):
    input_data = np.array([[n, p, k, ph, temperature, humidity, rainfall]])

    # Lưu ý: Nếu mô hình yêu cầu chuẩn hóa, bạn cần chuẩn hóa đầu vào bằng scaler đã lưu (nếu có)
    prediction = model.predict(input_data)
    st.success(f"✅ Cây trồng được đề xuất: **{prediction[0]}**")
