import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))
#print(data_dict.keys())
#print(data_dict)

# we have data and labels as lists we need to convert it into numpy array for ML algo
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

#print(data)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True,stratify=labels)
#Startify kya hai?
#if we have some inconsistency in so it may happen that only 1 type of data is trained and it will lead to overfitting and if we startify the y array then all the data sets are equally distributed in train and test mtlb ek incosistency train mein hai to test mein bhi 1 hoga agar startify nai karenge to ho sakta hai inconsistency wala khali test mein aaye and train mein khali chikna maal jaye
model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print(y_test)
print(y_predict)

score = accuracy_score(y_predict, y_test)

print(score)

f = open('model.p','wb')
pickle.dump({'model':model}, f)
f.close()