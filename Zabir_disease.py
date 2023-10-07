import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor

# Extracting the train data

disease_X_train= np.loadtxt('disease_X_train.txt')
disease_y_train= np.loadtxt('disease_y_train.txt')

# Extraction the test data

disease_X_test= np.loadtxt('disease_X_test.txt')
disease_y_test= np.loadtxt('disease_y_test.txt')

# Calculating the MSE for Baseline 

mean_disease_y_train=np.mean(disease_y_train)
mean_disease_y_train_array = np.full(len(disease_y_test), mean_disease_y_train)
mse_for_baseline = mean_squared_error(mean_disease_y_train_array, disease_y_test)

print()

print("The test set MSE for Baseline =",mse_for_baseline)

# Calculating the MSE for Linear Regression

Linear_model =LinearRegression(fit_intercept=True)
Linear_model.fit(disease_X_train,disease_y_train)
Linear_model_predict=Linear_model.predict(disease_X_test)
mse_for_LinearRegression = mean_squared_error(Linear_model_predict, disease_y_test)

print()

print("The test set MSE for Linear Regression =",mse_for_LinearRegression)

# Calculating the MSE for Decision  Tree

model_Decision_TreeRegressor = DecisionTreeRegressor(max_depth=5,min_samples_leaf=5,max_leaf_nodes=16) 
model_Decision_TreeRegressor.fit(disease_X_train,disease_y_train)
model_Decision_TreeRegressor_perdict=model_Decision_TreeRegressor.predict(disease_X_test)
mse_for_Decision_TreeRegressor = mean_squared_error(model_Decision_TreeRegressor_perdict, disease_y_test)

print()

print("The test set MSE for Decision Tree Regressor =",mse_for_Decision_TreeRegressor)

# Calculating the MSE for Random Forest

model_Random_Forest= RandomForestRegressor(n_estimators=500,max_depth=4,random_state=2)
model_Random_Forest.fit(disease_X_train,disease_y_train)
model_Random_Forest_predict=model_Random_Forest.predict(disease_X_test)
Mse_model_Random_Forest=mean_squared_error(model_Random_Forest_predict,disease_y_test)

print()

print("The test set MSE for Random Forest Regressor =",Mse_model_Random_Forest)

print()