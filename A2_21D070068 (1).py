import numpy as np
import pandas as pd
from sklearn import preprocessing
import streamlit as st
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

red=pd.read_csv("wine/winequality-red.csv",delimiter=";") #Reading the csv which has values seperated by ;
red['quality']=red['quality'].astype('int') #Convert target to int data type
scaler = preprocessing.StandardScaler() #Zero mean and unit variance normalization
scaler.fit(red.iloc[:, :-1]) #Applying the normalization
x_red=pd.DataFrame(scaler.transform(red.iloc[:, :-1])) #Normalizing all columns except the last column(quality) as it needs to remain discrete
y_red=pd.DataFrame(red['quality']) #Seperating the target column

param_grid = {'C': [0.01,0.1, 1, 10], #Dictionary with the values of C to validate
              'gamma': [0.01, 0.1, 1, 10]}  #Values of gamma to validate
X_train_red, X_vt, y_train_red, y_vt = train_test_split(x_red, y_red['quality'], test_size=0.3, random_state=42)  #Splitting data into train and val+test
X_test_red, X_val_red, y_test_red, y_val_red = train_test_split(X_vt, y_vt, test_size=0.5, random_state=42) #Splitting rest data into test and val
#Train=70%, Val=0.5*0.3=15%, Test=30-15=15%

mse_values_val = {gamma: [] for gamma in param_grid['gamma']}  #Dictionary for storing Val MSE for different params
mse_values_t = {gamma: [] for gamma in param_grid['gamma']}  #Dictionary for storing Train MSE for different params

for C in param_grid['C']: #Iterate over different values of C
    for gamma in param_grid['gamma']: #Iterate over different values of gamma
        svr = SVR(kernel='rbf', C=C, gamma=gamma)  #Define SVR with those hyper-params
        svr.fit(X_train_red, y_train_red)  #Train the SVR on training data
        y_pred_train = svr.predict(X_train_red)  #Make predictions on training data
        y_pred_val = svr.predict(X_val_red) #Make predictions on validation data
        mse_values_val[gamma].append(mean_squared_error(y_val_red, y_pred_val))  #Store the val MSE
        mse_values_t[gamma].append(mean_squared_error(y_train_red, y_pred_train))  #Store the train MSE

min_mse=99999999  #Initialising mse with a large value and change it every time the mse<current mse
c=10  #Initialising C with max value in param grid and change it every time the mse<current mse
gamma = 10  #Initialising gamma with max value in param grid and change it every time the mse<current mse

for g in param_grid['gamma']:  #Iterate over all values of gamma depth then update
    for i, mse_val in enumerate(mse_values_val[g]):  #Iterate over the dictionary storing the val MSE
        if mse_val < min_mse:  #If new val MSE>previous lowest val MSE
            min_mse=mse_val  #Update MSE
            c=param_grid['C'][i]  #Update C
            gamma=g  #Update gamma

print("Minimum MSE value:", min_mse)
print("Corresponding C:", c)
print("Corresponding gamma:", gamma)

best_svr_red = SVR(kernel='rbf', C=c, gamma=gamma)  #Define a SVR with the hyper-params which gave the lowest val MSE on val data
best_svr_red.fit(X_train_red, y_train_red)  #Train that SVR on train data

fixed_acidity = st.slider('fixed acidity', min_value=0.0, max_value=20.0, value=0.0, step=0.1) # Slider for fixed acidity
volatile_acidity = st.slider('volatile acidity', min_value=0.0, max_value=2.0, value=0.0, step=0.01) # Slider for volatile acidity
citric_acid = st.slider('citric acid', min_value=0.0, max_value=1.0, value=0.0, step=0.01) # Slider for citric acid
residual_sugar = st.slider('residual sugar', min_value=0.0, max_value=20.0, value=0.0, step=0.1) # Slider for residual sugar
chlorides = st.slider('chlorides', min_value=0.0, max_value=0.75, value=0.0, step=0.01) # Slider for chlorides
free_sulfur_dioxide = st.slider('free sulfur dioxide', min_value=0, max_value=100, value=0, step=1) # Slider for free sulfur dioxide
total_sulfur_dioxide = st.slider('total sulfur dioxide', min_value=0, max_value=300, value=0, step=1) # Slider for total sulfur dioxide
density = st.slider('density', min_value=0.0, max_value=1.5, value=0.0, step=0.001) # Slider for density
pH = st.slider('pH', min_value=1.0, max_value=4.0, value=0.0, step=0.01)# Slider for pH
sulphates = st.slider('sulphates', min_value=0.0, max_value=2.0, value=0.0, step=0.01) # Slider for sulphates
alcohol = st.slider('alcohol', min_value=0.0, max_value=20.0, value=0.0, step=0.1) # Slider for alcohol

if st.button('Make Prediction!'): #Button, which when clicked will start the prediction steps
    input={'fixed acidity': [fixed_acidity],'volatile acidity': [volatile_acidity],'citric acid': [citric_acid],'residual sugar': [residual_sugar],'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],'total sulfur dioxide': [total_sulfur_dioxide],'density': [density],'pH': [pH],'sulphates': [sulphates],'alcohol': [alcohol]}
    input = pd.DataFrame(input)
    scaler = preprocessing.StandardScaler() #Zero mean and unit variance normalization
    scaler.fit(input) #Applying the normalization
    input=pd.DataFrame(scaler.transform(input)) #Normalizing the features
    y_pred_test=best_svr_red.predict(input) #Make prediction on the input
    y_pred_test = np.clip(np.round(y_pred_test),3,8) #Clipping the prediction in 3-8 by rounding off as the Red Wine dataset's target values lie only in this range
    st.write("Quality = ", int(y_pred_test[0])) #Print the predicted quality
