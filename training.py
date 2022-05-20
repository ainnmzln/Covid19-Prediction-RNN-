# -*- coding: utf-8 -*-
"""

@author: ainnmzln
"""

import os,datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from module import ExploratoryDataAnalysis
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error

#%%
DATASET_TRAIN_PATH=os.path.join(os.getcwd(),'cases_malaysia_train.csv')
DATASET_TEST_PATH=os.path.join(os.getcwd(),'cases_malaysia_test.csv')
LOG_PATH=os.path.join(os.getcwd(),'log')
MODEL_PATH=os.path.join(os.getcwd(),'static','model.h5')

#%% Load data

train_df=pd.read_csv(DATASET_TRAIN_PATH)
test_df=pd.read_csv(DATASET_TEST_PATH)

x_train=train_df['cases_new'].values   #train the cases_new column only
x_test=test_df['cases_new'].values

# #%% Step 1. EDA

eda=ExploratoryDataAnalysis()

x_train=eda.remove_char(x_train)

x_train_numeric=eda.change_numeric(x_train)
x_test_numeric=eda.change_numeric(x_test)

x_train_imputed=eda.imputer(x_train_numeric)
x_test_imputed=eda.imputer(x_test_numeric)

x_train_floor=np.floor(x_train_imputed)
x_test_floor=np.floor(x_test_imputed)

x_train_floor=eda.scaler(x_train_floor)
x_test_floor=eda.scaler(x_test_floor)

#%% Step 2. Set up train and test dataset

window_size=30   

# Training dataset

x_train=[]
y_train=[]

for i in range(window_size,len(x_train_floor)): 
    x_train.append(x_train_floor[i-window_size:i,0])
    y_train.append(x_train_floor[i,0])

x_train=np.array(x_train)
y_train=np.array(y_train)

# Testing Dataset

temp=np.concatenate((x_train_floor,x_test_floor)) 
length_window=window_size+len(x_test_floor)
temp=temp[-length_window:]   

x_test=[]
y_test=[]

for i in range(window_size,len(temp)):
    x_test.append(temp[i-window_size:i,0])
    y_test.append(temp[i,0])

x_test=np.array(x_test)
y_test=np.array(y_test)

x_test=np.expand_dims(x_test,axis=-1)
x_train=np.expand_dims(x_train,axis=-1)

#%% Step 3. Model training 

model=Sequential()
model.add(LSTM(32,activation='tanh',
               return_sequences=(True),
               input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2)) 
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1,activation='relu'))
model.summary()

log_dir=os.path.join(LOG_PATH,
                     datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback=TensorBoard(log_dir=log_dir)

early_stopping=EarlyStopping(monitor='loss',patience=3)


model.compile(optimizer='adam', loss='mse', metrics='mse')

hist=model.fit(x_train,y_train,epochs=20,
               batch_size=128,
               callbacks=[tensorboard_callback,
                          early_stopping])

print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['mse'])
plt.show()

#%% Step 4. Model Evaluation

predicted=[]

for i in x_test:
    predicted.append(model.predict(np.expand_dims(i,axis=0)))
                                       
predicted=np.array(predicted)
predicted_floor=np.floor(predicted)

#%% Model Analysis

plt.figure()
plt.plot(predicted.reshape(len(predicted),1),color='g')   #reshape into 96,1
plt.plot(y_test,color='b')
plt.legend(['predicted','actual'])
plt.show()

# compute the score / mean_absolute_ percentage error

y_true=y_test
y_pred=predicted.reshape(len(predicted),1)

print((mean_absolute_error(y_true,y_pred)/(sum(abs(y_true))))*100)

#%%
model.save(MODEL_PATH)

