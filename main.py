import csv
from numpy.random.mtrand import uniform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import *
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

#data path
data_path = 'dataset/dataset_heart_disease.csv'

#read data
df = pd.read_csv(data_path)

#memisahkan data independent (fitur) dan dependent (class)
indep_data = df.iloc[:, 1:6]
dep_data = df.iloc[:, 6]

#Data Scaling untuk normalisasi data
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(indep_data))

#membagi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, dep_data, test_size=0.2)

# Train Menggunakan SVM
svm_clf = SVC()
svm_clf = svm_clf.fit(X_train, y_train)
predict_svm = svm_clf.predict(X_test)
print("SVM Prediction: ", accuracy_score(y_test, predict_svm))

# Train menggunakan KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform")
knn.fit(X_train, y_train)
predict_knn = knn.predict(X_test)
print("KNN Prediction: " , accuracy_score(y_test, predict_knn))

# Train menggunakan JST
jst = MLPClassifier(alpha=0.0001, hidden_layer_sizes=(15,), max_iter=2000)
jst.fit(X_train, y_train)
predict_jst = jst.predict(X_test)
print("JST Prediction: ", accuracy_score(y_test, predict_jst))
