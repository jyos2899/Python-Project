# -*- coding: utf-8 -*-

Importing Libraries
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns

"""Importing the DataSet "Epileptic Seizure Recognition.csv"."""

data = pd.read_csv('/content/Epileptic Seizure Recognition.csv')
data.shape

from google.colab import drive
drive.mount('/content/drive')

"""Separating the independent and dependent variable from the dataset."""

X = data.iloc[:,1:-1].values
X.shape
y = data.iloc[:,-1:].values
y[y>1] = 0
y.shape
y

"""Splitting the Variables into Training and Testing set """

X_train,X_test,y_train,y_test=train_test_split(X,y)
X_train.shape, X_test.shape

"""Scaling the feature variable using normalization. """

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_train_std
X_test_std

"""Composition of Dataset"""

values = data['y'].value_counts()
plt.figure(figsize=(7,7))
values.plot(kind='pie',fontsize=17, autopct='%.2f')
plt.legend(loc="best")
plt.show()

"""Classification of data"""

classifiers = [
    LogisticRegression(),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    MLPClassifier()]

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    print("="*30)
    print(name)
    print('****Results****')
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test) 
    print("Accuracy: ", acc)

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score
n = [5, 10, 20, 30, 40, 50, 100]
f1_scores = []
for i in n:
  pipe = [
        ('pca', PCA(n_components=i)),
        ('estimator', RandomForestClassifier())
    ]
  pipe = Pipeline(pipe)
  pipe.fit(X_train, y_train)
  f1_scores.append(f1_score(y_test, pipe.predict(X_test)))
ax = plt.axes()
ax.plot(n, f1_scores)
ax.set(xlabel='Number of Dimensions',
       ylabel='F1 Score')
ax.grid(True)

"""Dimensionality Reduction Using PCA(Principal component analysis)"""

cov_matrix = np.cov(X_train_std, rowvar=False)
egnvalues, egnvectors = eigh(cov_matrix)
total_egnvalues = sum(egnvalues)
var_exp = [(i/total_egnvalues) for i in sorted(egnvalues, reverse=True)]
egnpairs = [(np.abs(egnvalues[i]), egnvectors[:, i])
                for i in range(len(egnvalues))]
egnpairs.sort(key=lambda k: k[0], reverse=True)
projectionMatrix = []
for i in range(100):
  projectionMatrix.append((egnpairs[i][1][:,np.newaxis]))
projectionMatrix_1 = np.hstack(projectionMatrix)
projectionMatrix_1
X_train_pca = X_train_std.dot(projectionMatrix_1)
X_train_pca
X_test_pca = X_test_std.dot(projectionMatrix_1)

"""Development of model (Multilayer Perceptron)"""

#model
def denseBlock(dims,inp) :
    x = tf.keras.layers.BatchNormalization() (inp)
    x = tf.keras.layers.Dense(dims,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
    x = tf.keras.layers.Dropout(0.4) (x)
    x = tf.keras.layers.Dense(128,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
    return x

inp = tf.keras.layers.Input(shape=(100,),name='input')
x1 = denseBlock(256,inp)
x2 = denseBlock(512,inp)
x3 = denseBlock(1024,inp)
x4 = denseBlock(2048,inp)
x = tf.keras.layers.Concatenate()([x1,x2,x3,x4])
x = tf.keras.layers.Dense(128,activation=tf.keras.layers.LeakyReLU(0.2)) (x)
out = tf.keras.layers.Dense(1,activation='sigmoid',name='output') (x)

model = tf.keras.models.Model(inp,out)
model.summary()

tf.keras.utils.plot_model(model,show_shapes=True)

model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(1e-4),metrics=['accuracy'])
model.fit(X_train_pca,y_train,epochs=40,batch_size=128,validation_split=0.2)

model.evaluate(X_test_pca,y_test)

#On studying the pre existing research done in the field using the same dataset, it was found that the maximum accuracy achieved was 97%. With this model, the accuracy has been pushed to 97.84% which is significant considering the size of dataset we have used.
