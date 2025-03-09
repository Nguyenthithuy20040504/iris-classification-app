import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Tải dữ liệu Iris
iris = load_iris()
X, y = iris.data, iris.target

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Lưu mô hình vào file
with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Mô hình đã được huấn luyện và lưu thành công!")
