import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('lung cancer.csv')
dataset=pd.get_dummies(dataset,drop_first=True)
X = dataset.loc[:, ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
       'CHRONIC_DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
       'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
       'SWALLOWING_DIFFICULTY', 'CHEST_PAIN', 'GENDER_M']]            
y = dataset.loc[:, ['LUNG_CANCER']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classifier.predict(sc.transform([[21,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])))
print(y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
import pickle
file = open('result.pkl', 'wb')
pickle.dump(classifier, file)

