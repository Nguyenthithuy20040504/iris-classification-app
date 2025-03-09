import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_iris

# Tải dữ liệu Iris
iris = load_iris()
feature_names = iris.feature_names
class_names = iris.target_names

# Load mô hình đã huấn luyện
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Giao diện Streamlit
st.title("🌺 Ứng dụng phân loại hoa Iris")
st.write("Nhập thông tin các đặc trưng để dự đoán loại hoa.")

# Sidebar để nhập dữ liệu
st.sidebar.header("Thông tin đầu vào")
sepal_length = st.sidebar.slider(feature_names[0], 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider(feature_names[1], 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider(feature_names[2], 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider(feature_names[3], 0.1, 2.5, 1.0)

# Chuyển dữ liệu nhập vào thành numpy array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Dự đoán
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Hiển thị kết quả
st.subheader("Kết quả dự đoán:")
st.write(f"🔹 Loài hoa dự đoán: **{class_names[prediction[0]]}**")

st.subheader("Xác suất dự đoán:")
for i, class_name in enumerate(class_names):
    st.write(f"{class_name}: {prediction_proba[0][i]:.2%}")
