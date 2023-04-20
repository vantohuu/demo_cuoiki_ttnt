import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
data = pd.read_csv("Hotel Reservations.csv")

colunms = list(data.columns)

for c in colunms:
    data[c] = le.fit_transform(data[c])

X = data.drop('booking_status', axis=1)
y = data['booking_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)



from sklearn.metrics import accuracy_score   
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)