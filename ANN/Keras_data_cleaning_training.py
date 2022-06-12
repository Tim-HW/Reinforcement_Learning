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
from sklearn.metrics import explained_variance_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error,mean_squared_error
 

# get data
df = pd.read_csv(os.path.dirname(__file__)+"\\DATA\\kc_house_data.csv")
"""
# show if data are missing
"""
#print(df.isnull().sum())
#print(df.describe().transpose())
"""
# plot datas in dofferent forms
"""
#plt.figure(figsize=(10,6))
#sns.displot(df['price'])
#sns.countplot(df['bedrooms'])
#plt.show()
"""
# print the corrolations between two series
# that is fucking OP
"""
#print(df.corr()['bathrooms'].sort_values())
"""
# plot dots
"""
#sns.scatterplot(x='price',y='sqft_living',data=df)
"""
# plot variances
"""
#sns.boxplot(x='bedrooms',y='price',data=df)
#plt.show()

plt.figure(figsize=(12,8))
# print data from DF, hue = z component
#sns.scatterplot(x='lat',y='long',data=df,hue='price')
#plt.show()

# sort data along price
df.sort_values('price',ascending=False).head(20)
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]

"""
# print data from DF, hue = z component
# full broken plotter
"""
#sns.scatterplot(x='lat',y='long',data=non_top_1_perc,hue='price',alpha=0.2)
#plt.show()

#sns.boxenplot(x='waterfront',y='price',data=df)
#plt.show()

# get rid of ID column
df = df.drop('id', axis=1)
# transform object(strings) into Date values
df['date'] = pd.to_datetime(df['date'])
# extract year and transform into column
df['year'] = df['date'].apply(lambda date : date.year)
# extract month and transform into column
df['month'] = df['date'].apply(lambda date : date.month)
"""
#plot month by price
"""
#sns.boxplot(x='month',y='price',data=df)
#plt.show()
"""
#plot mean values of price per year
"""
#df.groupby('year').mean()['price'].plot()
#plt.show()

# get rid of date
df = df.drop('date',axis=1)

# show number of different values in the DF
#df['zipcode'].value_counts()
df = df.drop('zipcode',axis=1)

#print(df['yr_renovated'].value_counts())

# create features and outputs
x = df.drop('price',axis=1).values
y = df['price'].values
# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=101)
# scal Dataset
scaler = MinMaxScaler()
# fit the data into the scaler
scaler.fit(X_train)
# transform it
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test) 

# create NN model
model = Sequential()
# create layers
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
# init output
model.add(Dense(1))
# compile the model
model.compile(optimizer='adam',loss='mse')
# Training process with validation
model.fit(x=X_train,y=Y_train,
            validation_data=(X_test,Y_test),
            batch_size=128,epochs=400)

"""
# Plot dataframe of learning
"""
#df_loss = pd.DataFrame(model.history.history)
#df_loss.plot()
#plt.show()

predictions = model.predict(X_test)

error_RMS = mean_squared_error(Y_test,predictions)
erroe_abs = mean_absolute_error(Y_test,predictions)

explained_variance_score(Y_test,predictions)

plt.scatter(Y_test,predictions)
plt.plot(Y_test,Y_test)
plt.show()
