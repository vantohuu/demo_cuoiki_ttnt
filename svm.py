import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
data = pd.read_csv("Hotel Reservations.csv")

colunms = list(data.columns)

for c in colunms:
    data[c] = le.fit_transform(data[c])


X = data.drop('booking_status', axis=1)
y = data['booking_status']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)



from sklearn.model_selection import GridSearchCV

par = {
    'C': [0.1, 0.5, 1],
    'kernel': ['linear', 'polynomial', 'sigmoid'],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 0.5],
}
grid = GridSearchCV(SVC(), par)
grid.fit(X_train, y_train)
print(grid.best_params_)

svm = SVC(kernel= 'linear')
svm.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
y_pred = svm.predict(X_test)
print('Accuracy: %.3f' %accuracy_score(y_test, y_pred))
