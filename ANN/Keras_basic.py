from cgi import test
import os
from statistics import mode
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error,mean_squared_error

# get data
df = pd.read_csv(os.path.dirname(__file__)+"\\fake_reg.csv")
# show data
sns.pairplot(df)
plt.show()

# create features and output
X = df[['feature1','feature2']].values
Y = df['price'].values
# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

# init scaler
scaler = MinMaxScaler()
# feed the scaler
scaler.fit(X_train)
# update data
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

#model = Sequential([Dense(4,activation='relu'),
#                    Dense(2,activation='relu'),
#                    Dense(1,activation='relu')])

# init model
model = Sequential()
# dense is fully connected layers
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
# final output
model.add(Dense(1))
# compile it
model.compile(optimizer='rmsprop',loss='mse')
# fit data in it
model.fit(x=X_train,y=Y_train,epochs=250)
# get loss function
loss_df =pd.DataFrame(model.history.history)

# print loss function
#loss_df.plot()
#plt.show()

rms_eval  = model.evaluate(X_train,Y_train,verbose=0)
rms_train = model.evaluate(X_test,Y_test,verbose=0)

print("training RMS   : " + str(rms_train))
print("Evaluation RMS : " + str(rms_eval))

#######################################################
#               Further prediction
#######################################################

# get array of predicted price
test_prediction = model.predict(X_test)
# transform it into Pandas series
pred_df = pd.DataFrame(Y_test,columns=['Test Y'])
# reshape the serie
test_prediction = pd.Series(test_prediction.reshape(300,))
test_prediction = test_prediction.astype(np.float)
#print(pred_df.info())
#print(pred_df)
# concatenate both series
pred_df = pd.concat([pred_df,test_prediction],axis=1)
# redefine columns
pred_df.columns = ['Test True Y','Model Prediction']
pred_df['Test True Y'].astype(float)
# plot it
#sns.scatterplot(data=pred_df)
#plt.show()
# print mean absolute error
print(pred_df.info)

#print(mean_absolute_error(pred_df['Test True Y'],pred_df['Model Prediction']))
#print(mean_squared_error(pred_df['Test True Y'],pred_df['Model Prediction']))

##############################################
#                Inference
##############################################

inference = [[998,1000]]
inference = scaler.transform(inference)
print(model.predict(inference))

##############################################
#           Save weights
##############################################
model.save('my_model.h5')

later_model = load_model('my_model.h5')