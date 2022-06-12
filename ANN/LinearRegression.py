import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


#Series.keys() = Index
# converte to numpy array for operation
# Series1 + Serie2 = chelou
# series1.add(series1,0)

# dataframe

#serie = df.iloc[1]
#serie = df.loc['name']

# read CSV
df = pd.read_csv(os.path.dirname(__file__)+"\\Advertising.csv")
# drop column
x = df.drop('sales',axis=1)
# create serie with "Sales" 
y = df['sales']

# shuffle data
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.3,random_state=101)
# create a scaler
scaler = StandardScaler()
# 
scaler.fit(X_train)
# Scale data where mean value and standard deviation are known
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test) 


# create model
model = Ridge(alpha=1)
# fit data in it
model.fit(X_train,Y_train)
# create prediction
y_pred = model.predict(X_test)
# calulate the msqr of it
print(mean_squared_error(Y_test,y_pred))
