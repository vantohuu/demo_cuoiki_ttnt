import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
# Đọc file CSV và chuyển thành DataFrame
data = pd.read_csv("Hotel Reservations.csv")

# Chuyển các giá trị không phải số sang dạng số
colunms = list(data.columns)
for c in colunms:
    data[c] = le.fit_transform(data[c])

X = data.drop('booking_status', axis=1)
y = data['booking_status']

# Chia dữ liệu thành training set và testing set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Huấn luyện mô hình Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.5)
gb_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Dự đoán kết quả trên testing set và tính toán độ chính xác
y_pred = gb_clf.predict(X_test)
print('Accuracy: %.3f' %accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))