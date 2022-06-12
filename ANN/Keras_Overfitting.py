from cgi import test
import os
from statistics import mode
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error,mean_squared_error

# get data
df = pd.read_csv(os.path.dirname(__file__)+"\\DATA\\cancer_classification.csv")
# show data correlation
#df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')
# show heatmap 
sns.heatmap(df.corr())
plt.show()

# create features and output
X = df.drop('benign_0__mal_1',axis=1).values
Y = df['benign_0__mal_1'].values
# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=101)
# create scaler
scaler = MinMaxScaler()
# fit data in the scaler
scaler.fit(X_train)
# transform data
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)


model = Sequential()
# create layer
model.add(Dense(30,activation='relu'))
# disable random neuron to prevent over fitting
model.add(Dropout(0.5))
# create layer
model.add(Dense(15,activation='relu'))
# disable random neuron to prevent over fitting
model.add(Dropout(0.5))

# binary classification
model.add(Dense(1,activation='sigmoid'))
# compile the model
model.compile(loss='binary_crossentropy',optimizer='adam')
# create early stop event
early_Stop = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=25)
# train the model
model.fit(x=X_train,y=Y_train, epochs=600, validation_data=(X_test,Y_test),callbacks=early_Stop)
# show loss function
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()
# get prediction
predictions = model.predict_classes(X_test)
# print classification report
print(classification_report(Y_test,predictions))
# print confusion matrix
print(confusion_matrix(Y_test,predictions))