''' @author: Manogna Sunkara
Machine learning Final Project part 2
Dataset : Drug Classification '''


#Importing required libraries
import numpy as np
import pandas as pd #For reading dataset

#For plotting graphs
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.categorical import countplot
#for performance metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn import preprocessing
#Importing the data
data = pd.read_csv(r'drug200.csv')
data.shape
#We will check the attributes datatypes 
data.dtypes
#checking for Null values
data.isnull().sum()
#Count for each drug type
plt.figure(figsize = (9,5))
sns.countplot(data.Drug)
plt.show()
#Visualizing cholestrol and it's respective drug type count
plt.figure(figsize = (9,5))
sns.countplot(data= data, x="Drug",hue="Cholesterol")
plt.title('.....Cholesterol v/s Drug Type.....\n')
plt.show()
#Visualizing BP and it's respective drug type count
plt.figure(figsize = (9,5))
sns.countplot(data= data, x="Drug",hue="BP")
plt.title('.....BP v/s Drug Type.....\n')
plt.show()
#Using Label encoder on the Categorical attributes 
#Categorical attributes are : Sex, BP, Cholesterol and Drug
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data.Sex=le.fit_transform(data.Sex)
data.BP=le.fit_transform(data.BP)
data.Cholesterol=le.fit_transform(data.Cholesterol)
data.Drug=le.fit_transform(data.Drug)

#Showing the correlation between the attributes using Heat Map
sns.heatmap(data.corr(),annot=True,fmt='.1f',vmin=0.5, vmax=0.7 )
plt.show()
#Splitting the data into test and training datasets
X = data.drop(['Drug'], axis = 1)
y = data.Drug
from sklearn.model_selection import train_test_split
XTrain, XTest, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

print("____________________________________________________")

#Model 1 : Decision Tree
print("MODEL 1 : Decision Tree ")
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=5) #Create a DecisionTreeClassifier
dtc.fit(XTrain,y_train)
dtc_pred = dtc.predict(XTest)
#Calculating and printing accuracy of the Model #2
dtc_acc = accuracy_score(dtc_pred,y_test)
Confusion_Matrix = confusion_matrix(y_test, dtc_pred)
print(Confusion_Matrix)
#sns.heatmap(Confusion_Matrix, annot=True, annot_kws={"size": 16}, vmin=0.5, vmax=0.7)
#plt.show()
print('\n \n DecisionTreeClassifier accuracy:\t',dtc_acc)
print("____________________________________________________")
#model 2 : SVC
print("MODEL 2 : SVC ")
from sklearn.svm import SVC

svc_c = SVC()
svc_c.fit(XTrain, y_train)
svc_pred = svc_c.predict(XTest)
#Calculating and printing accuracy of the Model #1
svc_acc = accuracy_score(svc_pred,y_test)
print('\nSVC Classifier accuracy:\t',svc_acc)

#Confusion_Matrix = confusion_matrix(y_test, svc_pred)
#print(Confusion_Matrix)
print("____________________________________________________")
#Model 3 : GausianNB
print("MODEL 3 : Gausian Navive Bayes ")
from sklearn.naive_bayes import GaussianNB
gnb_c= GaussianNB()
gnb_c.fit(XTrain,y_train)
gnb_c_pred=gnb_c.predict(XTest)
gnb_acc = accuracy_score(gnb_c_pred, y_test)
print('Guasian Classifier accuracy:\t',gnb_acc)
print("____________________________________________________")
#Importing tensorflow and keras
import tensorflow as tf
from tensorflow import keras
#model 4 : Deep Neural Network
print("MODEL 4 : Deep Neural Network")
DNN_model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[5]),
keras.layers.BatchNormalization(),
keras.layers.Dense(100, activation="relu"),
keras.layers.BatchNormalization(),
keras.layers.Dense(100,activation="relu"),
keras.layers.Dense(5, activation="sigmoid")
])

DNN_model.summary()
#Vizualising DNN by plotting using Graphviz
keras.utils.plot_model(DNN_model, "DNN.png", show_shapes=True, rankdir="LR")
DNN_model.compile(loss="sparse_categorical_crossentropy",
optimizer=keras.optimizers.SGD(),
metrics=["accuracy","mse","mae"])
history = DNN_model.fit(XTrain, y_train, epochs=100, validation_data=(XTest, y_test))
#For testing and training: plotting accuracy, loss, mean absolute error, and mean squared error

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy Score')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy Score')
plt.title('Accuracy Score', loc='left', fontsize=16)
plt.xlabel("Epochs")
plt.ylabel('Accuracy Score')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='Training_Loss', c='blue')
plt.plot(history.history['val_loss'], label='Testing_Loss', c='yellow')
plt.title('Loss', loc='left', fontsize=16)
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(history.history['mae'], label='Training_MAE')
plt.plot(history.history['val_mae'], label='Testing_MAE')
plt.plot(history.history['mse'], label='Training_MSE', c='blue')
plt.plot(history.history['val_mse'], label='Testing_MSE', c='yellow')
plt.title('MAE and MSE', loc='left', fontsize=16)
plt.xlabel("Epochs")
plt.ylabel('Mean Absolute Error & Mean Squared Error')
plt.legend()
plt.show()

print("____________________________________________________")
#Accuracy Score for DNN and vizualising using Confusion matrix
seq_pred = np.argmax(DNN_model.predict(XTest), axis=1)
print('Accuracy Score of DNN :\t',accuracy_score(y_test, seq_pred))
Confusion_Matrix = confusion_matrix(y_test, seq_pred)
print("____________________________________________________")
print('DNN confusion matrix:\t',Confusion_Matrix)
sns.heatmap(Confusion_Matrix, annot=True, annot_kws={"size": 16}, vmin=0.5, vmax=0.7)
plt.show()