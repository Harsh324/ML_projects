import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
Data = datasets.load_diabetes()
#print(Data.keys())

Data_x = Data.data[:,np.newaxis,2]
#print(Data_x)

Data_x_train = Data_x[:-30]
Data_x_test = Data_x[-20:]

Data_y_train  = Data.target[:-30]
Data_y_test = Data.target[-20:]

Model = linear_model.LinearRegression()

Model.fit(Data_x_train,Data_y_train)

Var = Model.predict(Data_x_test)

print("mean squared error is : ",mean_squared_error(Data_y_test,Var))

print("Weights: ",Model.coef_)
print("Intercept: ",Model.intercept_)

plt.scatter(Data_x_test,Data_y_test)

plt.plot()