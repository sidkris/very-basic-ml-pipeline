import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle


training_data = pd.read_csv("storepurchasedata.csv")
print(training_data.describe())

X = training_data[["Age", "Salary"]].values
y = training_data["Purchased"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# using a knn classifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:,1]

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :\n {}".format(cm))

acc = accuracy_score(y_test, y_pred)
print("Accuracy % : {}".format(acc * 100))

new_prediction = knn.predict(sc.transform(np.array([[40, 50000]])))
print("new prediction : {}".format(new_prediction))

classification_model_file = "classifier.pickle"
pickle.dump(knn, open(classification_model_file, 'wb'))

scaler_file = "standard_scaler.pickle"
pickle.dump(sc, open(scaler_file, 'wb'))